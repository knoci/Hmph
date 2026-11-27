"""Three-layer local memory engine: short-term, bi-temporal versioned long-term, and working memory."""

from __future__ import annotations

import asyncio
import inspect
import logging
import os
import re
import time
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from hmph.lexical import tokenize
from hmph.local_memory.embeddings import Embedder, build_default_embedder
from hmph.local_memory.llm import (
    InvalidExtraction,
    LLMClient,
    OpenAICompatLLM,
    memory_locale,
    resolve_llm_config,
    resolve_locale_config,
)
from hmph.local_memory.reranker import (
    FeatureReranker,
    anchor_relative_dates,
    compute_actr_activation,
    parse_anchor_date,
)
from hmph.local_memory.store import LocalMemoryStore, StaleExtraction, normalize_namespace
from hmph.local_memory.types import (
    AmbiguityRecord,
    ConsolidationRecord,
    ConsolidationReport,
    CoreMemoryBlock,
    ExtractedChange,
    ForgetReceipt,
    ItemRecord,
    ItemStatus,
    MemoryEvent,
    MessageRecord,
    ProactiveSuggestion,
    SessionSummaryRecord,
    TurnEvidence,
    TurnResult,
)

logger = logging.getLogger(__name__)

HISTORY_CUE = re.compile(
    r"原来|当初|最早|一开始|之前|以前|先前|本来|原本|最初|怎么变|改过|历史|"
    r"\b(?:originally|initially|previously|used to|at first|before|history|changed from)\b",
    re.I,
)
CURRENT_CUE = re.compile(
    r"现在|目前|当前|如今|还(?:练|考|去|做|在|要|继续|坚持)|还在|现行|有哪些|什么计划|啥计划|"
    r"还记|记得|约[了的]?哪天|改到哪天|周几|礼拜几|星期几|几号|什么时候|吗[?？]?$|"
    r"\b(?:now|currently|still|active|what plan|which day|when is|do we still)\b",
    re.I,
)
PLAN_OVERVIEW_CUE = re.compile(
    r"(?:有哪些|有什么|还有什么|什么|哪些|全部|所有).{0,6}(?:计划|约定|安排|事项|待办)|"
    r"最近.{0,6}(?:安排|计划)|(?:计划|约定|安排).{0,6}(?:怎么样|状态|还在)|"
    r"\b(?:what plans|active plans|our plans|schedule|commitments)\b",
    re.I,
)
ALL_PLURAL_CUE = re.compile(r"都|全部|全都|一起|两[个项门件]|\b(?:both|all)\b", re.I)
MULTI_SESSION_CUE = re.compile(
    r"哪些|所有|全部|一共|总共|几次|多少|顺序|排序|从早到晚|先后|分别|去过哪|做过什么|参加过|"
    r"\b(?:order|earliest|latest|chronological|all of|how many|list all|compare|both|"
    r"(?:what|which)\s+(?:kinds?|types?|sorts?)\s+of\s+(?:places|activities|classes|groups|events|indoor|outdoor)|"
    r"(?:what|which)\s+(?:places|activities|classes|groups|museums|events))\b",
    re.I,
)
PREFERENCE_ADVICE_CUE = re.compile(
    r"建议|推荐|tips|怎么选|挑选|适合|主意|攻略|"
    r"\b(?:tips|recommend|recommendation|suggest|suggestion|advice|looking for|shopping for|what should i)\b",
    re.I,
)
ABSTRACT_INFERENCE_CUE = re.compile(
    r"爱好|兴趣|喜欢什么|适合什么|室内|户外|生活方式|可能|大概|偏好|"
    r"\b(?:indoor|outdoor|hobbies|interests|lifestyle|likely|potentially|prefer|preferred|favorite|"
    r"what\s+(?:could|might|would|can)\s+\w+\s+do|tips\s+for|recommend)\b",
    re.I,
)
TEMPORAL_ORIGIN_CUE = re.compile(
    r"最早|最初|一开始|哪一年|哪年|什么时候(?:开始|养|买|认识|领养)|"
    r"\b(?:which year|what year|first|originally|initially|earliest|when did\s+\w+\s+(?:first|adopt|start|begin|get|meet))\b",
    re.I,
)
TRIVIAL_BACKCHANNEL_RE = re.compile(
    r"^(?:哈哈+|嘿嘿+|呵呵+|嗯+|哦+|噢+|好[的呀啦呗]?|行[的吧呀]?|收到|明白|了解|"
    r"谢谢[你啦]?|多谢|早安|早上好|晚安|拜拜|再见|"
    r"ok|okay|k|yes|yep|yeah|no|nope|sure|got it|thanks|thank you|thx|lol|haha|gm|gn|"
    r"[?？!！。.,，~～…👍❤️😂🤣😊🙏]+)$",
    re.I,
)


def is_trivial_backchannel(text: str) -> bool:
    """Return True only for unambiguous short acknowledgments/greetings with zero memory payload."""
    cleaned = (text or "").strip()
    if not cleaned or len(cleaned) > 12:
        return False
    return bool(TRIVIAL_BACKCHANNEL_RE.match(cleaned))

ACTION_TO_STATUS: dict[str, ItemStatus] = {
    "create": "active",
    "reschedule": "active",
    "update": "active",
    "pause": "paused",
    "resume": "active",
    "complete": "completed",
    "cancel": "cancelled",
    "retract": "retracted",
}


def reciprocal_rank_fusion(rankings: list[list[str]], *, limit: int = 24) -> list[str]:
    scores: dict[str, float] = defaultdict(float)
    for ranking in rankings:
        seen: set[str] = set()
        for rank, item_id in enumerate(ranking, start=1):
            if item_id not in seen:
                scores[item_id] += 1.0 / (60.0 + rank)
                seen.add(item_id)
    return sorted(scores, key=lambda key: (-scores[key], key))[:limit]


def classify_query_intent(query: str) -> str:
    asks_history = bool(HISTORY_CUE.search(query))
    asks_current = bool(CURRENT_CUE.search(query)) or bool(PLAN_OVERVIEW_CUE.search(query))
    if asks_history and asks_current:
        return "both"
    if asks_history:
        return "history"
    if asks_current or "?" in query or "？" in query:
        return "current"
    return "casual"


def _anchor_quote(quote: str, original: str) -> str:
    cleaned = (quote or "").strip()
    if cleaned and cleaned in original:
        return cleaned
    stripped = cleaned.strip(" \"'“”‘’。！？!?，,")
    if stripped and stripped in original:
        return stripped
    return original.strip()


def _item_directly_matched(query: str, item: ItemRecord) -> bool:
    q_low = query.lower()
    if item.title.lower() in q_low:
        return True
    for ent in item.entities:
        if len(ent) >= 2 and ent.lower() in q_low:
            return True
    q_tokens = set(tokenize(query))
    if not q_tokens:
        return False
    item_text = " ".join(
        [
            item.title,
            " ".join(item.entities),
            *(v.summary for v in item.versions),
            *(v.source_quote for v in item.versions),
        ]
    )
    i_tokens = set(tokenize(item_text))
    overlap = q_tokens & i_tokens
    # Require >=2 bigram/word overlap, or 1 overlap if query is very short.
    return len(overlap) >= 2 or (len(q_tokens) <= 3 and len(overlap) >= 1)


def _chrono_sort_key(ts_str: str | None, tie: str = "") -> tuple[int, str, str]:
    """True chronological sort key handling both ISO dates and English dates like '1:10 pm on 27 March, 2023'."""
    dt = parse_anchor_date(ts_str)
    if dt is not None:
        return (0, dt.strftime("%Y-%m-%d"), str(ts_str or "") + "|" + tie)
    return (1, str(ts_str or ""), tie)


def _format_version_line(v, *, zh: bool = False) -> str:
    parts = [f"[{v.change_type} -> {v.status}]: {v.summary}"]
    if v.effective_time_text:
        parts.append(f"({'时间' if zh else 'Time'}: {v.effective_time_text})")
    if v.valid_from and v.valid_to:
        parts.append(f"({'有效区间' if zh else 'Validity'}: {v.valid_from[:10]}~{v.valid_to[:10]})")
    if v.condition_text:
        parts.append(f"({'条件' if zh else 'Condition'}: {v.condition_text})")
    return " ".join(parts)


class LocalMemoryEngine:
    """Standalone single-process SQLite memory engine with background structured extraction."""

    def __init__(
        self,
        db_path: Path | str,
        *,
        llm: LLMClient,
        embedder: Embedder | None = None,
        timezone_name: str = "Asia/Hong_Kong",
        locale: str = "auto",
        recent_turn_limit: int = 8,
        enable_ppr: bool = True,
        enable_two_hop: bool = True,
        enable_reranker: bool = True,
        enable_listwise_rerank: bool = True,
        enable_dedup: bool = True,
        auto_consolidate_every: int = 12,
        auto_summarize_turns: int = 6,
        session_idle_minutes: float = 45.0,
        enable_trivial_gate: bool = True,
    ):
        self.embedder = embedder if embedder is not None else build_default_embedder()
        self.store = LocalMemoryStore(db_path, embedder=self.embedder)
        self.llm = llm
        self.timezone_name = timezone_name
        self.locale = locale
        self.recent_turn_limit = recent_turn_limit
        self.enable_ppr = enable_ppr
        self.enable_two_hop = enable_two_hop
        self.enable_reranker = enable_reranker
        self.enable_listwise_rerank = enable_listwise_rerank
        self.enable_dedup = enable_dedup
        self.auto_consolidate_every = max(0, int(auto_consolidate_every))
        self.auto_summarize_turns = max(0, int(auto_summarize_turns))
        self.session_idle_minutes = max(0.0, float(session_idle_minutes))
        self.enable_trivial_gate = enable_trivial_gate
        self.reranker = FeatureReranker()
        self._bg_tasks: set[asyncio.Task] = set()
        self._extract_lock = asyncio.Lock()
        self._extraction_tasks: dict[str, asyncio.Task] = {}
        self._consolidate_lock = asyncio.Lock()
        self._mutations_since_consolidation: dict[str, int] = defaultdict(int)
        self._last_active_session: dict[str, str] = {}
        self._event_subscribers: dict[str, set[asyncio.Queue[MemoryEvent]]] = defaultdict(set)
        self.last_extraction_usage: dict[str, int] | None = None
        self.extraction_errors: list[str] = []
        self.skipped_trivial_extractions: int = 0
        self._ttft_history_ms: list[float] = []
        self._total_history_ms: list[float] = []

    @classmethod
    def from_env(
        cls,
        db_path: Path | str,
        *,
        embedder: Embedder | None = None,
        temperature: float = 0.3,
    ) -> LocalMemoryEngine:
        api_key, base_url, model, tz = resolve_llm_config()
        locale = resolve_locale_config()
        idle_env = os.getenv("HMPH_SESSION_IDLE_MINUTES", "").strip()
        idle_mins = float(idle_env) if idle_env else 45.0
        llm = OpenAICompatLLM(
            api_key,
            base_url=base_url,
            model=model,
            temperature=temperature,
            locale=locale,
        )
        return cls(
            db_path,
            llm=llm,
            embedder=embedder,
            timezone_name=tz,
            locale=locale,
            session_idle_minutes=idle_mins,
        )

    def _now_iso(self) -> str:
        return datetime.now(ZoneInfo(self.timezone_name)).isoformat(timespec="seconds")

    # ------------------------------------------------------------------
    # Public Chat & Ingestion Interface
    # ------------------------------------------------------------------

    async def chat(
        self,
        session_id: str,
        user_text: str,
        *,
        namespace: str = "default",
        now_iso: str | None = None,
        await_extraction: bool = False,
    ) -> TurnResult:
        """Persist raw utterance, assemble 3-layer working memory, reply, and extract in background."""
        ns = normalize_namespace(namespace)
        ts = now_iso or self._now_iso()
        recent_before = self.store.get_session_messages(
            session_id, namespace=ns, limit=self.recent_turn_limit
        )

        # 1. Immediately persist user message + FTS5 + dense vector before reply.
        user_msg = self.store.add_message(
            session_id,
            "user",
            user_text,
            namespace=ns,
            timezone_name=self.timezone_name,
            created_at=ts,
        )

        self._schedule_extraction(ns)

        # 2. Assemble short-term + long-term + working memory with mention gating.
        t0 = time.perf_counter()
        evidence = self._assemble_working_memory(
            session_id=session_id,
            query=user_text,
            current_user_message_id=user_msg.id,
            namespace=ns,
        )
        retrieval_ms = (time.perf_counter() - t0) * 1000.0
        evidence = evidence.model_copy(update={"retrieval_ms": round(retrieval_ms, 2)})

        # 3. Generate companion reply using narrated context (no raw UUID JSON).
        t1 = time.perf_counter()
        reply, usage = await self.llm.generate_reply(
            user_message=user_text,
            recent_messages=recent_before,
            narrated_context=evidence.narrated_context,
            now_iso=ts,
            timezone_name=self.timezone_name,
        )
        generation_ms = (time.perf_counter() - t1) * 1000.0

        # 4. Persist assistant reply and record provenance links for cascade forgetting.
        assistant_msg = self.store.add_message(
            session_id,
            "assistant",
            reply,
            namespace=ns,
            timezone_name=self.timezone_name,
            created_at=self._now_iso() if now_iso is None else ts,
            reply_to_message_id=user_msg.id,
            index_for_retrieval=False,
        )
        self.store.record_turn_evidences(
            assistant_msg.id,
            item_ids=evidence.used_item_ids,
            source_message_ids=evidence.used_source_message_ids,
            ambiguity_ids=evidence.ambiguity_ids,
        )

        if await_extraction:
            await self.flush()

        return TurnResult(
            session_id=session_id,
            user_message_id=user_msg.id,
            assistant_message_id=assistant_msg.id,
            reply=reply,
            evidence=evidence,
            generation_ms=round(generation_ms, 2),
            usage=usage,
        )

    async def ingest_user_message(
        self,
        session_id: str,
        user_text: str,
        *,
        namespace: str = "default",
        now_iso: str | None = None,
        await_extraction: bool = True,
    ) -> MessageRecord:
        """Direct ingestion entry point for offline benchmark evaluation & headless remember API."""
        ns = normalize_namespace(namespace)
        ts = now_iso or self._now_iso()
        user_msg = self.store.add_message(
            session_id,
            "user",
            user_text,
            namespace=ns,
            timezone_name=self.timezone_name,
            created_at=ts,
        )
        self._schedule_extraction(ns)
        if await_extraction:
            await self.flush()
            return self.store.get_message(user_msg.id) or user_msg
        return user_msg

    async def ingest_session(
        self,
        session_id: str,
        turns: list[dict],
        *,
        namespace: str = "default",
        extract_state: bool = True,
    ) -> dict:
        """Ingest a full session of turns (for long-conversation evaluation & batch indexing).

        Each turn dict contains: external_id, role ('user'|'assistant'), content, timestamp, speaker (optional).
        Returns {"ingested_messages": int, "writer_calls": int, "writer_tokens": int}.
        """
        ns = normalize_namespace(namespace)
        async with self._extract_lock:
            if extract_state and not await self._consume_pending_extractions(ns):
                raise RuntimeError("An earlier memory extraction failed; retry before importing state")
            return await self._ingest_session(session_id, turns, ns, extract_state)

    async def _ingest_session(
        self, session_id: str, turns: list[dict], ns: str, extract_state: bool
    ) -> dict:
        extract_fn = getattr(self.llm, "extract_session_changes", None)
        has_extractor = extract_state and callable(extract_fn)
        stored_msgs: list[tuple[int, MessageRecord, dict]] = []
        for idx, turn in enumerate(turns):
            role = "user" if turn.get("role", "user") == "user" else "assistant"
            speaker = turn.get("speaker")
            text = str(turn["content"]).strip()
            formatted = f"{speaker}: {text}" if speaker and not text.startswith(f"{speaker}:") else text
            msg = self.store.add_message(
                session_id,
                "user",
                formatted,
                namespace=ns,
                timezone_name=self.timezone_name,
                created_at=str(turn.get("timestamp") or self._now_iso()),
                index_for_retrieval=True,
                external_id=turn.get("external_id"),
                extract_state=has_extractor,
            )
            stored_msgs.append((idx, msg, {"turn_index": idx, "speaker": speaker or role, "time": msg.created_at, "text": formatted}))

        if not has_extractor or not stored_msgs:
            return {"ingested_messages": len(stored_msgs), "writer_calls": 0, "writer_tokens": 0}

        session_blob = "\n".join(payload["text"] for _, _, payload in stored_msgs)
        candidates = self._select_extraction_candidates(session_blob[:3000], namespace=ns, limit=18)

        resp, usage = await extract_fn(
            session_turns=[payload for _, _, payload in stored_msgs],
            candidate_items=candidates,
            timezone_name=self.timezone_name,
        )
        changes_by_source = defaultdict(list)
        for change in resp.changes:
            idx = change.source_turn_index
            if idx is None or not 0 <= idx < len(stored_msgs):
                idx = len(stored_msgs) - 1
            changes_by_source[idx].append(change)
        for idx, msg, _ in stored_msgs:
            changes = changes_by_source[idx]
            operations = []
            if changes:
                operations = self._validate_and_plan_operations(
                    msg.content, changes, self.store.list_items(namespace=ns), namespace=ns
                )
            self.store.commit_extraction(msg.id, operations, created_at=msg.created_at)

        tokens = int((usage or {}).get("total_tokens", 0))
        return {"ingested_messages": len(stored_msgs), "writer_calls": 1, "writer_tokens": tokens}

    async def evaluate_query(
        self,
        query: str,
        *,
        namespace: str = "default",
        mode: str = "local_state",
        oracle_external_ids: list[str] | tuple[str, ...] = (),
        now_iso: str | None = None,
        max_sources: int = 12,
        max_context_bytes: int = 12000,
        max_output_tokens: int = 256,
    ) -> dict:
        """Controlled benchmark reader supporting 'raw_only', 'local_state', and 'oracle' arms."""
        ns = normalize_namespace(namespace)
        ts = now_iso or self._now_iso()
        t0 = time.perf_counter()

        # Shared raw retriever (exact ONNX cosine + FTS5 BM25 RRF fusion).
        msg_sem_scored = self.store.search_messages_semantic_scored(
            query, namespace=ns, limit=30
        )
        msg_sem = [mid for mid, _ in msg_sem_scored]
        top_sem_score = msg_sem_scored[0][1] if msg_sem_scored else 0.0
        msg_fts = self.store.search_messages_fts(query, namespace=ns, limit=30)
        raw_ranked = reciprocal_rank_fusion([msg_sem, msg_fts], limit=24)

        selected_msgs: list[MessageRecord] = []
        seen_msg_ids: set[str] = set()
        state_sections: list[str] = []
        used_item_ids: list[str] = []
        used_consolidation_ids: list[str] = []
        intent = classify_query_intent(query)

        def add_msg(msg_id: str) -> None:
            if msg_id in seen_msg_ids or len(selected_msgs) >= max_sources:
                return
            rec = self.store.get_message(msg_id)
            if rec is not None:
                selected_msgs.append(rec)
                seen_msg_ids.add(msg_id)

        if mode == "oracle":
            for mid in self.store.get_message_ids_by_external(oracle_external_ids, namespace=ns):
                add_msg(mid)
            source_heading = "[Original Conversation Excerpts]"
            source_messages = sorted(selected_msgs, key=lambda m: m.created_at)
        elif mode == "raw_only":
            # Diversity-first filling: reserve up to 8 slots for distinct top hits across sessions,
            # then fill remaining slots with adjacent confirmations in the same session.
            primary_limit = min(max_sources, max(8, max_sources - 4))
            primary_ids: list[str] = []
            for mid in raw_ranked:
                if len(selected_msgs) >= primary_limit:
                    break
                before = len(selected_msgs)
                add_msg(mid)
                if len(selected_msgs) > before:
                    primary_ids.append(mid)
            for mid in primary_ids:
                if len(selected_msgs) >= max_sources:
                    break
                for nb in self.store.get_neighbor_user_messages(mid, radius=1):
                    if len(selected_msgs) < max_sources:
                        add_msg(nb.id)
            for mid in raw_ranked:
                if len(selected_msgs) >= max_sources:
                    break
                add_msg(mid)
            source_heading = "[Retrieved Original Conversation Excerpts (chronological)]"
            source_messages = sorted(selected_msgs, key=lambda m: m.created_at)
        elif mode == "local_state":
            # Frontier multi-channel retrieval:
            # (1) Dense ONNX + FTS5 on items & raw messages
            # Spread relevance through entity, item, and message links without another model call.
            # (3) Adaptive 2-Hop Query Decomposition only for multi-session / ordering / enumeration queries
            # (4) Multi-feature + Session-Diversity Reranking + Deterministic Relative-Time Anchoring
            is_multi_session = bool(MULTI_SESSION_CUE.search(query))
            is_abstract_inference = bool(ABSTRACT_INFERENCE_CUE.search(query))
            is_temporal_origin = bool(TEMPORAL_ORIGIN_CUE.search(query))
            item_sem = self.store.search_items_semantic(
                query, namespace=ns, limit=20, min_score=0.35
            )
            item_fts = self.store.search_items_fts(query, namespace=ns, limit=20)
            fused_items = reciprocal_rank_fusion([item_sem, item_fts], limit=10)
            one_hop = self.store.expand_one_hop_items(fused_items[:4], limit=4)

            ppr_item_ids: list[str] = []
            ppr_msg_scores: dict[str, float] = {}
            if self.enable_ppr:
                ppr_items, ppr_msgs = self.store.run_bipartite_ppr(
                    query=query,
                    seed_item_ids=fused_items,
                    seed_message_ids=raw_ranked[:12],
                    namespace=ns,
                    top_items=10,
                    top_messages=16,
                )
                ppr_item_ids = [iid for iid, _ in ppr_items]
                ppr_msg_scores = dict(ppr_msgs)

            # Adaptive 2-Hop Bridge Planning (3-condition gate):
            # 1) multi-session enumeration/ordering, OR
            # 2) mathematically low first-pass dense confidence (< 0.52), OR
            # 3) abstract category / preference / indirect inference query on a long history.
            msg_total = self.store.count_user_messages(namespace=ns)
            low_confidence_gap = (
                top_sem_score < 0.52 and msg_total >= 40 and intent != "casual"
            )
            abstract_gap = is_abstract_inference and msg_total >= 40
            should_consider_two_hop = is_multi_session or low_confidence_gap or abstract_gap
            second_hop_used = False
            second_hop_tokens = 0
            extra_msg_sem: list[str] = []
            extra_msg_fts: list[str] = []
            extra_item_sem: list[str] = []
            if self.enable_two_hop and should_consider_two_hop:
                plan_fn = getattr(self.llm, "plan_second_hop", None)
                if callable(plan_fn):
                    first_pass_snips = []
                    for mid in raw_ranked[:8]:
                        m = self.store.get_message(mid)
                        if m is not None:
                            first_pass_snips.append(f"({m.created_at}) {m.content[:180]}")
                    try:
                        hop_plan, hop_usage = await plan_fn(
                            question=query, first_pass_snippets=first_pass_snips
                        )
                        second_hop_tokens = int((hop_usage or {}).get("total_tokens", 0))
                        if hop_plan.get("need_second_hop"):
                            second_hop_used = True
                            for subq in hop_plan.get("subqueries", [])[:3]:
                                extra_msg_sem.extend(
                                    self.store.search_messages_semantic(
                                        subq, namespace=ns, limit=18
                                    )
                                )
                                extra_msg_fts.extend(
                                    self.store.search_messages_fts(
                                        subq, namespace=ns, limit=18
                                    )
                                )
                                extra_item_sem.extend(
                                    self.store.search_items_semantic(
                                        subq, namespace=ns, limit=10, min_score=0.35
                                    )
                                )
                            if self.enable_ppr and (extra_msg_sem or extra_item_sem):
                                ppr2_items, ppr2_msgs = self.store.run_bipartite_ppr(
                                    query=query + " " + " ".join(hop_plan.get("subqueries", [])),
                                    seed_item_ids=list(dict.fromkeys([*fused_items, *extra_item_sem]))[:12],
                                    seed_message_ids=list(dict.fromkeys([*raw_ranked[:10], *extra_msg_sem[:12]]))[:18],
                                    namespace=ns,
                                    top_items=12,
                                    top_messages=20,
                                )
                                for iid, _ in ppr2_items:
                                    if iid not in ppr_item_ids:
                                        ppr_item_ids.append(iid)
                                for mid, sc in ppr2_msgs:
                                    ppr_msg_scores[mid] = max(ppr_msg_scores.get(mid, 0.0), sc)
                    except Exception as exc:
                        logger.warning("Adaptive second-hop planner failed gracefully: %s", exc)

            candidate_item_ids = reciprocal_rank_fusion(
                [fused_items, one_hop, ppr_item_ids, extra_item_sem], limit=10
            )
            recalled_items: list[ItemRecord] = []
            for iid in candidate_item_ids:
                it = self.store.get_item(iid)
                if it is not None:
                    recalled_items.append(it)
            recalled_set = {it.id for it in recalled_items}
            for it in self.store.list_items(namespace=ns):
                if it.id not in recalled_set and _item_directly_matched(query, it):
                    recalled_items.insert(0, it)
                    recalled_set.add(it.id)
            ledger_cap = 6 if is_multi_session else 4
            recalled_items = recalled_items[:ledger_cap]
            used_item_ids = [it.id for it in recalled_items]

            if recalled_items:
                lines = [
                    "[Structured State Evolution Ledger (relevant tracked states; "
                    "for topics not listed here, rely on the timestamped excerpts below)]"
                ]
                for it in recalled_items:
                    cur = it.current
                    desc = cur.summary if cur else it.title
                    time_part = f" | time: {cur.effective_time_text}" if (cur and cur.effective_time_text) else ""
                    cond_part = f" | condition: {cur.condition_text}" if (cur and cur.condition_text) else ""
                    ep_tag = " (belief)" if it.epistemic_type == "inferred_belief" else ""
                    lines.append(f"- [{it.status}]{ep_tag} {it.title}: {desc}{time_part}{cond_part}")
                    if len(it.versions) > 1:
                        lines.append(
                            "  Evolution chain: "
                            + " -> ".join(_format_version_line(v) for v in it.versions)
                        )
                        if intent == "current":
                            lines.append(
                                "  (Note: earlier superseded/retracted versions are no longer currently valid.)"
                            )
                state_sections.append("\n".join(lines))

            is_pref_advice = bool(PREFERENCE_ADVICE_CUE.search(query)) or is_abstract_inference
            cons_hits = self.store.search_consolidations(
                query, namespace=ns, limit=4, include_top_if_empty=is_pref_advice
            )
            entity_cons = self.store.search_consolidations_by_entities(
                query, namespace=ns, limit=6
            )
            merged_cons: list[ConsolidationRecord] = []
            seen_cons_ids: set[str] = set()
            # Keep semantic+FTS5 top dossiers FIRST so generic ubiquitous entities never evict exact matches.
            for c in [*cons_hits, *entity_cons]:
                if c.id not in seen_cons_ids:
                    merged_cons.append(c)
                    seen_cons_ids.add(c.id)
            cons_hits = merged_cons[: (6 if (is_pref_advice or is_multi_session) else 5)]
            if cons_hits:
                used_consolidation_ids = [c.id for c in cons_hits]
                clines = ["[Consolidated Entity Dossiers & Preferences (grounded in history)]"]
                for cons in cons_hits:
                    clines.append(f"- {cons.topic}: {cons.summary}")
                state_sections.append("\n".join(clines))

            # Channel 5: Coreference-resolved Session Summaries + Entity-Scoped Recall
            # Ensures asking about any person/relationship surfaces ALL their cross-session episodes
            # (even when turns omit abstract umbrella words like "indoor").
            sem_ss = self.store.search_session_summaries(
                query, namespace=ns, limit=10 if is_multi_session else 6
            )
            ent_ss = self.store.search_session_summaries_by_entities(
                query, namespace=ns, limit=12
            )
            ss_hits: list[SessionSummaryRecord] = []
            seen_ss: set[str] = set()
            # Interleave entity-IDF and semantic summaries, and for temporal-origin queries
            # ("which year", "first", "originally") always include the earliest matching sessions
            # chronologically so later mentions never drown out the original event!
            for idx_pair in range(max(len(ent_ss), len(sem_ss))):
                if idx_pair < len(ent_ss) and ent_ss[idx_pair].id not in seen_ss:
                    ss_hits.append(ent_ss[idx_pair])
                    seen_ss.add(ent_ss[idx_pair].id)
                if idx_pair < len(sem_ss) and sem_ss[idx_pair].id not in seen_ss:
                    ss_hits.append(sem_ss[idx_pair])
                    seen_ss.add(sem_ss[idx_pair].id)
            if is_temporal_origin:
                all_ns_ss = self.store.list_session_summaries(namespace=ns)
                q_ents = self.store.match_known_entities(query, namespace=ns)
                if q_ents:
                    matching_early = [
                        s
                        for s in all_ns_ss
                        if any(
                            self.store._entity_present_in_text(
                                e, f"{s.summary} {' '.join(s.facts)} {' '.join(s.entities)}".lower()
                            )
                            for e in q_ents
                        )
                    ]
                else:
                    matching_early = all_ns_ss
                earliest_ent = sorted(
                    matching_early or all_ns_ss,
                    key=lambda s: _chrono_sort_key(s.session_date, s.session_id),
                )[:2]
                for early_s in reversed(earliest_ent):
                    if early_s.id not in seen_ss:
                        ss_hits.insert(0, early_s)
                        seen_ss.add(early_s.id)
                    else:
                        ss_hits = [early_s, *[s for s in ss_hits if s.id != early_s.id]]
            if second_hop_used and extra_msg_sem:
                ss_by_sid = {s.session_id: s for s in self.store.list_session_summaries(namespace=ns)}
                for mid in extra_msg_sem[:10]:
                    m_rec = self.store.get_message(mid)
                    if m_rec and m_rec.session_id in ss_by_sid:
                        cand_s = ss_by_sid[m_rec.session_id]
                        if cand_s.id not in seen_ss:
                            ss_hits.append(cand_s)
                            seen_ss.add(cand_s.id)

            # Build unified candidate message pool combining first-pass, PPR, second-hop, and summary provenance.
            all_sem = list(dict.fromkeys([*msg_sem, *extra_msg_sem]))
            all_fts = list(dict.fromkeys([*msg_fts, *extra_msg_fts]))
            sem_rank_map = {mid: r for r, mid in enumerate(all_sem, start=1)}
            fts_rank_map = {mid: r for r, mid in enumerate(all_fts, start=1)}
            ss_msg_seeds: list[str] = []
            for ss in ss_hits[:6]:
                for mid in ss.source_message_ids[:3]:
                    if mid not in ss_msg_seeds:
                        ss_msg_seeds.append(mid)
            pool_ids = list(
                dict.fromkeys(
                    [*raw_ranked, *extra_msg_sem, *extra_msg_fts, *ppr_msg_scores.keys(), *ss_msg_seeds]
                )
            )[:48]

            if self.enable_reranker and pool_ids:
                cand_dicts = []
                for mid in pool_ids:
                    rec = self.store.get_message(mid)
                    if rec is not None:
                        cand_dicts.append(
                            {
                                "id": rec.id,
                                "session_id": rec.session_id,
                                "content": rec.content,
                                "created_at": rec.created_at,
                                "sem_rank": sem_rank_map.get(rec.id),
                                "fts_rank": fts_rank_map.get(rec.id),
                                "ppr_score": ppr_msg_scores.get(rec.id, 0.0),
                            }
                        )
                ranked_msg_ids = self.reranker.rerank_messages(
                    query,
                    cand_dicts,
                    intent=intent,
                    limit=max_sources * 2,
                    diverse_sessions=(is_multi_session or second_hop_used),
                )
            else:
                ranked_msg_ids = reciprocal_rank_fusion(
                    [all_sem, all_fts, list(ppr_msg_scores.keys())], limit=max_sources * 2
                )

            # Optionally rerank the evidence as a set.
            # jointly selects the sharpest session summaries + raw excerpts from Top-24 candidates.
            selected_summaries: list[SessionSummaryRecord] = []
            listwise_used = False
            listwise_fn = getattr(self.llm, "rerank_evidence_listwise", None)
            should_listwise = (
                self.enable_listwise_rerank
                and callable(listwise_fn)
                and msg_total >= 40
                and (is_multi_session or is_abstract_inference or second_hop_used or bool(ss_hits))
            )
            if should_listwise:
                lw_cands: list[dict] = []
                ss_lw_cap = 14 if (is_multi_session or is_abstract_inference) else 10
                msg_lw_cap = 24 - min(ss_lw_cap, len(ss_hits))
                for ss in ss_hits[:ss_lw_cap]:
                    facts_txt = " ; ".join(ss.facts[:8])
                    lw_cands.append(
                        {
                            "type": "summary",
                            "ref": ss,
                            "date": ss.session_date,
                            "text": f"{ss.summary} | Facts: {facts_txt}",
                        }
                    )
                for mid in ranked_msg_ids[:msg_lw_cap]:
                    rec = self.store.get_message(mid)
                    if rec is not None:
                        lw_cands.append(
                            {
                                "type": "msg",
                                "ref": rec,
                                "date": rec.created_at,
                                "text": rec.content,
                            }
                        )
                if len(lw_cands) >= 4:
                    try:
                        picked_idxs, _ = await listwise_fn(
                            query=query,
                            candidates=lw_cands,
                            top_k=min(14, len(lw_cands)),
                        )
                        if picked_idxs:
                            listwise_used = True
                            for pi in picked_idxs:
                                cand = lw_cands[pi]
                                if cand["type"] == "summary":
                                    ss_obj: SessionSummaryRecord = cand["ref"]
                                    if ss_obj not in selected_summaries:
                                        selected_summaries.append(ss_obj)
                                    # Also pull the top-matching original turns from this selected session!
                                    q_toks_lw = set(tokenize(query))
                                    sess_msgs = [
                                        self.store.get_message(m_id)
                                        for m_id in ss_obj.source_message_ids
                                    ]
                                    valid_sm = [m for m in sess_msgs if m is not None]
                                    valid_sm.sort(
                                        key=lambda m: -len(q_toks_lw & set(tokenize(m.content)))
                                    )
                                    for m in valid_sm[:2]:
                                        add_msg(m.id)
                                else:
                                    m_obj: MessageRecord = cand["ref"]
                                    add_msg(m_obj.id)
                                    if not is_multi_session and len(selected_msgs) < max_sources:
                                        for nb in self.store.get_neighbor_user_messages(m_obj.id, radius=1):
                                            add_msg(nb.id)
                    except Exception as exc:
                        logger.warning("Listwise evidence reranker failed gracefully: %s", exc)

            if not selected_summaries and ss_hits:
                selected_summaries = ss_hits[: (8 if is_multi_session else 4)]
            elif (is_multi_session or is_temporal_origin) and ss_hits:
                # Guarantee top IDF-ranked / earliest-origin summaries aren't over-pruned by listwise selection.
                seen_sel = {s.id for s in selected_summaries}
                cap_preserve = 8 if is_multi_session else 4
                for ss in ss_hits[:cap_preserve]:
                    if ss.id not in seen_sel and len(selected_summaries) < cap_preserve:
                        selected_summaries.append(ss)
                        seen_sel.add(ss.id)

            if selected_summaries:
                ss_lines = ["[Disambiguated Session Summaries & Anchored Atomic Facts (chronological)]"]
                for ss in sorted(
                    selected_summaries,
                    key=lambda x: _chrono_sort_key(x.session_date, x.session_id),
                ):
                    ss_lines.append(f"- ({ss.session_date} | {ss.session_id}) {ss.summary}")
                    for f in ss.facts[:6]:
                        ss_lines.append(f"    • {f}")
                state_sections.append("\n".join(ss_lines))

            # Guarantee Top-4 exact first-pass raw hits + adjacent session continuity for
            # single-hop questions (top-1 gets radius=3 so focused multi-turn discussions retain
            # their opening context; top-2..4 get radius=1).
            if not is_multi_session:
                for idx_r, mid in enumerate(raw_ranked[:4]):
                    if len(selected_msgs) >= max_sources:
                        break
                    add_msg(mid)
                    rad = 3 if idx_r == 0 else 1
                    nb_cap = min(max_sources, 9 if idx_r == 0 else 8)
                    for nb in self.store.get_neighbor_user_messages(mid, radius=rad):
                        if len(selected_msgs) < nb_cap:
                            add_msg(nb.id)
            else:
                for mid in raw_ranked[:2]:
                    add_msg(mid)

            # Pull verified provenance source messages for tracked ledger items:
            # for multi-session ordering/enumeration, cover up to 6 recalled items (1 per item)
            # so every tracked event contributes its timestamped original turn.
            prov_item_cap = min(6, len(recalled_items)) if is_multi_session else 2
            item_provenance_budget = min(
                len(selected_msgs) + (prov_item_cap if is_multi_session else 2),
                max_sources,
            )
            for it in recalled_items[:prov_item_cap]:
                if is_multi_session:
                    versions_to_pull = list(reversed(it.versions))[:1]
                elif len(it.versions) >= 2 and intent in {"history", "both"}:
                    versions_to_pull = [it.versions[0], it.versions[-1]]
                else:
                    versions_to_pull = list(reversed(it.versions))[:2]
                for ver in versions_to_pull:
                    if len(selected_msgs) >= item_provenance_budget:
                        break
                    add_msg(ver.source_message_id)
                if len(selected_msgs) >= item_provenance_budget:
                    break

            for mid in ranked_msg_ids:
                if len(selected_msgs) >= max_sources:
                    break
                add_msg(mid)
            for mid in raw_ranked:
                if len(selected_msgs) >= max_sources:
                    break
                add_msg(mid)

            source_heading = "[Verified Original Excerpts (chronological)]"
            if selected_msgs:
                ph = ",".join("?" for _ in selected_msgs)
                order = dict(self.store._conn.execute(
                    f"SELECT message_id, sequence FROM message_ingest_order WHERE message_id IN ({ph})",
                    [m.id for m in selected_msgs],
                ).fetchall())
            else:
                order = {}
            source_messages = sorted(
                selected_msgs,
                key=lambda m: (*_chrono_sort_key(m.created_at)[:2], order.get(m.id, 0)),
            )
        else:
            raise ValueError(f"Unsupported evaluation mode: {mode}")

        budget = max(0, max_context_bytes)
        context_bytes = "\n\n".join(state_sections).encode("utf-8")[:budget]
        context_text = context_bytes.decode("utf-8", errors="ignore")
        remaining = budget - len(context_text.encode("utf-8"))
        source_prefix = ("\n\n" if context_text else "") + source_heading + "\n"

        used_message_ids: list[str] = []
        truncated_message_ids: list[str] = []
        for message in source_messages:
            if not message.content:
                continue
            hint = anchor_relative_dates(message.content, message.created_at) if mode == "local_state" else ""
            prefix = f"{source_prefix}- ({message.created_at}){hint} "
            available = remaining - len(prefix.encode("utf-8"))
            if available <= 0:
                break
            content = message.content.encode("utf-8")
            excerpt = content[:available].decode("utf-8", errors="ignore")
            if not excerpt:
                break
            context_text += prefix + excerpt
            if len(content) > available:
                truncated_message_ids.append(message.id)
                break
            used_message_ids.append(message.id)
            remaining = available - len(content)
            source_prefix = "\n"

        retrieval_ms = (time.perf_counter() - t0) * 1000.0
        t1 = time.perf_counter()
        reply, usage = await self.llm.answer_from_context(
            question=query,
            context_text=context_text,
            now_iso=ts,
            timezone_name=self.timezone_name,
            max_tokens=max_output_tokens,
        )
        generation_ms = (time.perf_counter() - t1) * 1000.0

        ext_map = self.store.get_external_ids([*used_message_ids, *truncated_message_ids])
        included_ids = set(used_message_ids) | set(truncated_message_ids)
        return {
            "mode": mode,
            "intent": intent,
            "reply": reply,
            "used_item_ids": used_item_ids,
            "candidate_message_ids": [m.id for m in selected_msgs],
            "used_message_ids": used_message_ids,
            "used_external_ids": [ext_map[mid] for mid in used_message_ids if mid in ext_map],
            "truncated_message_ids": truncated_message_ids,
            "truncated_external_ids": [ext_map[mid] for mid in truncated_message_ids if mid in ext_map],
            "omitted_message_ids": [m.id for m in selected_msgs if m.id not in included_ids],
            "context_bytes": len(context_text.encode("utf-8")),
            "retrieval_ms": round(retrieval_ms, 2),
            "generation_ms": round(generation_ms, 2),
            "total_seconds": round((retrieval_ms + generation_ms) / 1000.0, 3),
            "top_sem_score": round(top_sem_score, 4),
            "second_hop_used": second_hop_used if mode == "local_state" else False,
            "second_hop_tokens": second_hop_tokens if mode == "local_state" else 0,
            "listwise_used": listwise_used if mode == "local_state" else False,
            "used_consolidation_ids": used_consolidation_ids,
            "used_session_summary_ids": [s.id for s in selected_summaries]
            if mode == "local_state"
            else [],
            "usage": usage,
        }

    # ------------------------------------------------------------------
    # Real-Time Memory Event Bus (for AIRI Plugins / Live2D Reactions)
    # ------------------------------------------------------------------

    def subscribe_events(self, *, namespace: str = "default") -> asyncio.Queue[MemoryEvent]:
        ns = normalize_namespace(namespace)
        q: asyncio.Queue[MemoryEvent] = asyncio.Queue(maxsize=64)
        self._event_subscribers[ns].add(q)
        return q

    def unsubscribe_events(self, q: asyncio.Queue[MemoryEvent], *, namespace: str = "default") -> None:
        ns = normalize_namespace(namespace)
        self._event_subscribers[ns].discard(q)

    def _publish_event(self, event_type: str, *, namespace: str = "default", payload: dict | None = None) -> None:
        ns = normalize_namespace(namespace)
        subs = list(self._event_subscribers.get(ns, ())) + list(self._event_subscribers.get("*", ()))
        if not subs:
            return
        evt = MemoryEvent(
            event_type=event_type,  # type: ignore[arg-type]
            namespace=ns,
            timestamp=self._now_iso(),
            payload=payload or {},
        )
        for q in subs:
            try:
                q.put_nowait(evt)
            except asyncio.QueueFull:
                pass

    # ------------------------------------------------------------------
    # Session Summarization (Coreference-Resolved Atomic Facts)
    # ------------------------------------------------------------------

    async def summarize_session(
        self, session_id: str, *, namespace: str = "default"
    ) -> SessionSummaryRecord | None:
        ns = normalize_namespace(namespace)
        msgs = self.store.get_session_messages(session_id, namespace=ns, limit=200)
        if not msgs:
            return None
        summarize_fn = getattr(self.llm, "summarize_session_facts", None)
        if not callable(summarize_fn):
            return None
        session_date = msgs[0].created_at
        turns = [
            {"speaker": m.role, "time": m.created_at, "text": m.content}
            for m in msgs
        ]
        result, _ = await summarize_fn(
            session_id=session_id,
            session_date=session_date,
            turns=turns,
            timezone_name=self.timezone_name,
        )
        summary_text = str(result.get("summary") or "").strip()
        facts = list(result.get("facts") or [])
        entities = list(result.get("entities") or [])
        if not summary_text and not facts:
            return None
        rec = self.store.upsert_session_summary(
            session_id,
            namespace=ns,
            session_date=session_date,
            summary=summary_text or (facts[0] if facts else session_id),
            facts=facts,
            entities=entities,
            source_message_ids=[m.id for m in msgs],
            updated_at=self._now_iso(),
        )
        self._publish_event(
            "session_summarized",
            namespace=ns,
            payload={"session_id": session_id, "fact_count": len(rec.facts)},
        )
        return rec

    async def backfill_session_summaries(
        self,
        *,
        namespace: str = "default",
        concurrency: int = 3,
        force: bool = False,
    ) -> dict:
        """Build coreference-resolved session summaries for all unsummarized sessions in a namespace."""
        ns = normalize_namespace(namespace)
        sessions = self.store.list_sessions(namespace=ns, limit=500)
        existing_sids = (
            set()
            if force
            else {s.session_id for s in self.store.list_session_summaries(namespace=ns)}
        )
        todo = [s["session_id"] for s in sessions if s["session_id"] not in existing_sids]
        if not todo:
            return {
                "namespace": ns,
                "summarized": 0,
                "total_summaries": len(self.store.list_session_summaries(namespace=ns)),
            }

        sem = asyncio.Semaphore(max(1, concurrency))
        created = 0

        async def _worker(sid: str) -> None:
            nonlocal created
            async with sem:
                rec = await self.summarize_session(sid, namespace=ns)
                if rec is not None:
                    created += 1

        await asyncio.gather(*(_worker(sid) for sid in todo), return_exceptions=True)
        return {
            "namespace": ns,
            "summarized": created,
            "total_summaries": len(self.store.list_session_summaries(namespace=ns)),
        }

    def list_session_summaries(self, *, namespace: str = "default") -> list[SessionSummaryRecord]:
        return self.store.list_session_summaries(namespace=namespace)

    # ------------------------------------------------------------------
    # Core memory blocks and proactive check-ins
    # ------------------------------------------------------------------

    def get_core_memory_blocks(self, *, namespace: str = "default") -> list[CoreMemoryBlock]:
        return self.store.get_core_memory_blocks(namespace=namespace)

    def set_core_memory_block(
        self,
        block_key: str,
        content: str,
        *,
        namespace: str = "default",
        source_item_ids: list[str] | tuple[str, ...] = (),
    ) -> CoreMemoryBlock:
        blk = self.store.upsert_core_memory_block(
            block_key,
            content,
            namespace=namespace,
            source_item_ids=source_item_ids,
            updated_at=self._now_iso(),
        )
        self._publish_event(
            "core_memory_updated",
            namespace=blk.namespace,
            payload={"block_key": blk.block_key},
        )
        return blk

    def delete_core_memory_block(self, block_key: str, *, namespace: str = "default") -> bool:
        deleted = self.store.delete_core_memory_block(block_key, namespace=namespace)
        if deleted:
            self._publish_event(
                "core_memory_updated",
                namespace=namespace,
                payload={"block_key": block_key, "deleted": True},
            )
        return deleted

    def get_proactive_suggestion(
        self,
        *,
        namespace: str = "default",
        cooldown_hours: float = 12.0,
    ) -> ProactiveSuggestion:
        """Boundary-safe proactive companion initiator for AIRI idle/heartbeat loops.

        Strictly excludes paused, cancelled, retracted, and completed items so AIRI NEVER nags.
        """
        ns = normalize_namespace(namespace)
        all_items = self.store.list_items(namespace=ns)
        paused_count = sum(1 for it in all_items if it.status == "paused")

        # 1. Open ambiguities take priority (polite clarification).
        open_ambs = self.store.list_open_ambiguities(namespace=ns, limit=1)
        if open_ambs:
            amb = open_ambs[0]
            titles = []
            for cid in amb.candidate_item_ids:
                it = self.store.get_item(cid)
                if it is not None and it.namespace == ns:
                    titles.append(it.title)
            cand_str = " / ".join(titles) if titles else "earlier items"
            return ProactiveSuggestion(
                namespace=ns,
                should_initiate=True,
                reason="open_ambiguity",
                topic=cand_str,
                suggested_cue=f"Gently clarify which item the user meant in '{amb.source_quote}' ({cand_str}).",
                excluded_paused_count=paused_count,
            )

        # 2. Active plans past cooldown window (NEVER paused/closed items!).
        now_dt = datetime.now(UTC)
        for it in all_items:
            if it.status != "active" or it.kind != "plan":
                continue
            ref_ts = it.last_mentioned_at or it.updated_at
            elapsed_hours = cooldown_hours + 1.0
            try:
                dt = datetime.fromisoformat(str(ref_ts).replace("Z", "+00:00"))
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=UTC)
                elapsed_hours = (now_dt - dt).total_seconds() / 3600.0
            except ValueError:
                pass
            if elapsed_hours >= cooldown_hours:
                cur = it.current
                summary = cur.summary if cur else it.title
                time_hint = f" ({cur.effective_time_text})" if (cur and cur.effective_time_text) else ""
                return ProactiveSuggestion(
                    namespace=ns,
                    should_initiate=True,
                    reason="active_plan_checkin",
                    item_id=it.id,
                    topic=it.title,
                    suggested_cue=f"Lightly check in on active plan '{it.title}': {summary}{time_hint}.",
                    excluded_paused_count=paused_count,
                )

        return ProactiveSuggestion(
            namespace=ns,
            should_initiate=False,
            reason="none",
            excluded_paused_count=paused_count,
        )

    async def consolidate(self, *, namespace: str = "default") -> ConsolidationReport:
        """Sleep Consolidation: deduplicate items, synthesize vector-indexed insights, decay stale episodes."""
        ns = normalize_namespace(namespace)
        await self.flush()
        return await self._run_consolidation(namespace=ns)

    async def _run_consolidation(self, *, namespace: str = "default") -> ConsolidationReport:
        ns = normalize_namespace(namespace)
        async with self._consolidate_lock:
            merged = self.store.merge_near_duplicate_items(namespace=ns, threshold=0.86)
            items = [
                it
                for it in self.store.list_items(namespace=ns)
                if it.status in {"active", "paused"}
            ]
            item_map = {it.id: it for it in items}
            existing = [
                {"topic": c.topic, "summary": c.summary}
                for c in self.store.list_consolidations(namespace=ns)
            ]
            saved: list[ConsolidationRecord] = self.store.list_consolidations(namespace=ns)
            usage: dict[str, int] | None = None

            consolidate_fn = getattr(self.llm, "consolidate_insights", None)
            if callable(consolidate_fn) and items:
                kwargs: dict = {
                    "items": items,
                    "existing_consolidations": existing,
                    "timezone_name": self.timezone_name,
                }
                try:
                    sig = inspect.signature(consolidate_fn)
                    if "session_summaries" in sig.parameters:
                        kwargs["session_summaries"] = self.store.list_session_summaries(
                            namespace=ns
                        )
                except (TypeError, ValueError):
                    pass
                raw_insights, usage = await consolidate_fn(**kwargs)
                validated: list[dict] = []
                core_profile_text = ""
                core_source_items: list[str] = []
                for entry in raw_insights:
                    if not core_profile_text and entry.get("_core_profile"):
                        core_profile_text = str(entry["_core_profile"]).strip()
                    valid_item_ids = [
                        iid for iid in (entry.get("source_item_ids") or []) if iid in item_map
                    ]
                    if not valid_item_ids:
                        continue
                    core_source_items.extend(valid_item_ids)
                    src_msg_ids: list[str] = []
                    for iid in valid_item_ids:
                        for ver in item_map[iid].versions:
                            if ver.source_message_id not in src_msg_ids:
                                src_msg_ids.append(ver.source_message_id)
                    validated.append(
                        {
                            "topic": entry["topic"],
                            "summary": entry["summary"],
                            "source_item_ids": valid_item_ids,
                            "source_message_ids": src_msg_ids,
                            "confidence": entry.get("confidence", 0.85),
                        }
                    )
                saved = self.store.replace_consolidations(
                    validated, namespace=ns, updated_at=self._now_iso()
                )
                if not core_profile_text and saved:
                    core_profile_text = "；".join(
                        f"{c.topic}：{c.summary[:90]}" for c in saved[:3]
                    )
                if core_profile_text:
                    self.store.upsert_core_memory_block(
                        "user_profile",
                        core_profile_text[:600],
                        namespace=ns,
                        source_item_ids=list(dict.fromkeys(core_source_items)),
                        updated_at=self._now_iso(),
                    )

            decayed = self.store.decay_stale_salience(namespace=ns)
            self._mutations_since_consolidation[ns] = 0
            self._publish_event(
                "consolidation_completed",
                namespace=ns,
                payload={
                    "merged_duplicates": merged,
                    "consolidations_count": len(saved),
                    "decayed_items": decayed,
                },
            )
            return ConsolidationReport(
                namespace=ns,
                merged_duplicates=merged,
                decayed_items=decayed,
                consolidations=tuple(saved),
                usage=usage,
            )

    def list_consolidations(self, *, namespace: str = "default") -> list[ConsolidationRecord]:
        return self.store.list_consolidations(namespace=namespace)

    def list_namespaces(self) -> list[str]:
        return self.store.list_namespaces()

    def export_graph_data(self, *, namespace: str = "default") -> dict:
        return self.store.export_graph_data(namespace=namespace)

    def rebuild_fts_anchors(self) -> int:
        return self.store.rebuild_fts_anchors()

    async def flush(self) -> None:
        """Wait until all scheduled background extraction & consolidation tasks complete."""
        while self._bg_tasks:
            tasks = list(self._bg_tasks)
            await asyncio.gather(*tasks, return_exceptions=True)

    async def close(self) -> None:
        """Flush in-flight extractions and cleanly close the SQLite database."""
        await self.flush()
        aclose = getattr(self.llm, "aclose", None)
        if callable(aclose):
            await aclose()
        self.store.close()

    def list_items(
        self, *, namespace: str = "default", status: ItemStatus | None = None
    ) -> list[ItemRecord]:
        return self.store.list_items(namespace=namespace, status=status)

    def list_open_ambiguities(self, *, namespace: str = "default") -> list[AmbiguityRecord]:
        return self.store.list_open_ambiguities(namespace=namespace)

    def recall_context(
        self,
        query: str,
        *,
        namespace: str = "default",
        session_id: str = "mcp-default",
        as_of: str | None = None,
    ) -> TurnEvidence:
        """Public read-only memory recall API (used by AIRI Gateway, MCP hmph_recall & SDK callers)."""
        t0 = time.perf_counter()
        ev = self._assemble_working_memory(
            session_id=session_id,
            query=query,
            current_user_message_id="",
            namespace=namespace,
            as_of=as_of,
        )
        return ev.model_copy(update={"retrieval_ms": round((time.perf_counter() - t0) * 1000.0, 2)})

    def get_items_as_of(
        self,
        as_of: str,
        *,
        namespace: str = "default",
        include_closed: bool = False,
    ) -> list[ItemRecord]:
        return self.store.get_items_as_of(
            as_of, namespace=namespace, include_closed=include_closed
        )

    def export_namespace(
        self, *, namespace: str = "default", fmt: str = "json"
    ) -> dict | str:
        if fmt.strip().lower() in {"md", "markdown"}:
            return self.store.export_namespace_markdown(namespace=namespace)
        return self.store.export_namespace_data(namespace=namespace)

    def import_namespace(
        self, payload: dict, *, target_namespace: str | None = None
    ) -> dict[str, int]:
        return self.store.import_namespace_data(payload, target_namespace=target_namespace)

    async def recover_pending_extractions(self, *, namespace: str | None = None) -> int:
        """Crash-safe startup recovery: resume any `extracted = 0` user turns left behind."""
        namespaces = (
            [normalize_namespace(namespace)] if namespace else self.list_namespaces()
        )
        scheduled = 0
        for ns in namespaces:
            pending = self.store.get_pending_messages(namespace=ns)
            if pending:
                self._schedule_extraction(ns)
                scheduled += len(pending)
        return scheduled

    def resolve_episode_session_id(
        self,
        base_session_id: str,
        *,
        namespace: str = "default",
        now_iso: str | None = None,
        explicit: bool = False,
    ) -> str:
        """Start a new session after an idle gap for clients without explicit session IDs."""
        clean_base = (base_session_id or "").strip() or f"airi-{normalize_namespace(namespace)}"
        if explicit or self.session_idle_minutes <= 0.0:
            return clean_base
        ns = normalize_namespace(namespace)
        latest_sid, latest_ts, ep_no = self.store.get_latest_session_activity(
            clean_base, namespace=ns
        )
        if latest_sid is None or latest_ts is None:
            return clean_base
        ts = now_iso or self._now_iso()
        try:
            dt_prev = datetime.fromisoformat(latest_ts.replace("Z", "+00:00"))
            dt_now = datetime.fromisoformat(ts.replace("Z", "+00:00"))
            if dt_prev.tzinfo is None:
                dt_prev = dt_prev.replace(tzinfo=UTC)
            if dt_now.tzinfo is None:
                dt_now = dt_now.replace(tzinfo=UTC)
            idle_mins = (dt_now - dt_prev).total_seconds() / 60.0
            if idle_mins >= self.session_idle_minutes:
                # Track previous episode so `_maybe_schedule_auto_summary` summarizes it even across restarts!
                self._last_active_session.setdefault(ns, latest_sid)
                return f"{clean_base}#ep{ep_no + 1}"
        except ValueError:
            pass
        return latest_sid

    def list_sessions(self, *, namespace: str = "default", limit: int = 40) -> list[dict]:
        return self.store.list_sessions(namespace=namespace, limit=limit)

    def get_session_messages(
        self, session_id: str, *, namespace: str = "default", limit: int = 100
    ) -> list[MessageRecord]:
        return self.store.get_session_messages(session_id, namespace=namespace, limit=limit)

    def latency_stats(self) -> dict:
        import math
        from statistics import median

        def summarize(values: list[float]) -> dict:
            if not values:
                return {"count": 0, "p50_ms": None, "p95_ms": None, "last_ms": None}
            ordered = sorted(values)
            p95_idx = max(0, math.ceil(0.95 * len(ordered)) - 1)
            return {
                "count": len(values),
                "p50_ms": round(median(ordered), 1),
                "p95_ms": round(ordered[p95_idx], 1),
                "last_ms": round(values[-1], 1),
            }

        return {
            "ttft": summarize(self._ttft_history_ms),
            "total": summarize(self._total_history_ms),
        }

    async def toggle_item_pause(
        self, item_id: str, *, session_id: str, namespace: str = "default"
    ) -> ItemRecord | None:
        """Pause an active item or resume a paused item with a sourced explicit user record."""
        ns = normalize_namespace(namespace)
        await self.flush()
        item = self.store.get_item(item_id)
        if item is None or item.namespace != ns or item.status not in {"active", "paused"}:
            return item if (item is not None and item.namespace == ns) else None
        ts = self._now_iso()
        zh = memory_locale(self.locale, item.title) == "zh"
        if item.status == "active":
            text = f"手动暂停事项：{item.title}，暂时不用提醒。" if zh else f"Manually paused: {item.title}. Hold reminders."
            action = "pause"
            new_status: ItemStatus = "paused"
            summary = f"{item.title}已手动暂停" if zh else f"{item.title} was paused"
            eff_time = None
        else:
            last_active = next(
                (v for v in reversed(item.versions) if v.status == "active" and v.effective_time_text),
                None,
            )
            eff_time = last_active.effective_time_text if last_active else None
            text = f"手动恢复事项：{item.title}。" if zh else f"Manually resumed: {item.title}."
            action = "resume"
            new_status = "active"
            summary = f"恢复{item.title}" if zh else f"{item.title} was resumed"
            if eff_time:
                summary += f"（延续时间：{eff_time}）" if zh else f" (schedule: {eff_time})"
        src_msg = self.store.add_message(
            session_id,
            "user",
            text,
            namespace=ns,
            timezone_name=self.timezone_name,
            created_at=ts,
        )
        self.store.commit_extraction(
            src_msg.id,
            [
                {
                    "kind": "add_version",
                    "item_id": item.id,
                    "title": item.title,
                    "status": new_status,
                    "change_type": action,
                    "summary": summary,
                    "effective_time_text": eff_time,
                    "source_quote": text,
                }
            ],
            created_at=ts,
        )
        updated_item = self.store.get_item(item.id)
        if updated_item is not None:
            self._publish_event(
                "item_updated",
                namespace=ns,
                payload={
                    "item_id": updated_item.id,
                    "title": updated_item.title,
                    "status": updated_item.status,
                    "action": action,
                },
            )
        return updated_item

    # ------------------------------------------------------------------
    # AIRI / OpenAI-Compatible Persona-Preserving Memory Middleware
    # ------------------------------------------------------------------

    @staticmethod
    def extract_latest_user_text(messages: list[dict]) -> str:
        """Extract latest user utterance text from OpenAI messages[] (handles string & multimodal list)."""
        for msg in reversed(messages):
            if msg.get("role") != "user":
                continue
            content = msg.get("content")
            if isinstance(content, str):
                text = content.strip()
                if text:
                    return text
            elif isinstance(content, list):
                parts = [
                    str(part.get("text", "")).strip()
                    for part in content
                    if isinstance(part, dict) and part.get("type") == "text"
                ]
                joined = "\n".join(p for p in parts if p).strip()
                if joined:
                    return joined
        return ""

    @staticmethod
    def inject_memory_into_messages(
        messages: list[dict],
        narrated_context: str,
        *,
        now_iso: str,
        timezone_name: str,
        locale: str = "en",
    ) -> list[dict]:
        """Extend the caller's system prompt with recalled context."""
        cloned = [dict(m) for m in messages]
        if not narrated_context.strip():
            return cloned

        if locale == "zh":
            guidance = (
                f"当前时间：{now_iso}（{timezone_name}）\n"
                "用角色自己的语气自然参考以下记忆，保持当前对话的语言。"
                "区分最新状态和历史记录。暂停的计划等用户问起再提。"
                "可以联系证据清楚的线索回答，缺少相关记录时如实说明。"
            )
        else:
            guidance = (
                f"Current time: {now_iso} ({timezone_name})\n"
                "Use the recalled memory naturally in your character voice and the current conversation's language. "
                "Distinguish current state from superseded versions. Mention paused plans when asked. "
                "Connect clearly supported clues, and say when there is no relevant record."
            )
        memory_block = f"\n\n<hmph_memory>\n{guidance}\n{narrated_context.strip()}\n</hmph_memory>"

        for idx, msg in enumerate(cloned):
            if msg.get("role") == "system":
                existing = msg.get("content")
                if isinstance(existing, str):
                    cloned[idx] = {**msg, "content": existing.rstrip() + memory_block}
                elif isinstance(existing, list):
                    cloned[idx] = {
                        **msg,
                        "content": [*existing, {"type": "text", "text": memory_block}],
                    }
                return cloned

        cloned.insert(0, {"role": "system", "content": memory_block.strip()})
        return cloned

    def prepare_gateway_turn(
        self,
        messages: list[dict],
        *,
        namespace: str = "default",
        session_id: str = "airi-default",
        memory_mode: str = "auto",
        now_iso: str | None = None,
    ) -> tuple[list[dict], TurnEvidence | None, MessageRecord | None, list[MessageRecord], str]:
        """Intercept an OpenAI-compatible request from AIRI, recall namespace memory, and inject cleanly."""
        ns = normalize_namespace(namespace)
        ts = now_iso or self._now_iso()
        mode = (memory_mode or "auto").strip().lower()
        if mode == "bypass":
            return [dict(m) for m in messages], None, None, [], ts

        user_text = self.extract_latest_user_text(messages)
        if not user_text:
            return [dict(m) for m in messages], None, None, [], ts

        recent_before = self.store.get_session_messages(
            session_id, namespace=ns, limit=self.recent_turn_limit
        )
        user_msg: MessageRecord | None = None
        if mode == "auto":
            # AIRI retry idempotency: reuse identical user turn if retried within 20s.
            dup = self.store.find_recent_duplicate_message(
                session_id, user_text, namespace=ns, within_seconds=20
            )
            if dup is not None:
                user_msg = dup
            else:
                user_msg = self.store.add_message(
                    session_id,
                    "user",
                    user_text,
                    namespace=ns,
                    timezone_name=self.timezone_name,
                    created_at=ts,
                )
            if not user_msg.extracted:
                self._schedule_extraction(ns)

        t0 = time.perf_counter()
        evidence = self._assemble_working_memory(
            session_id=session_id,
            query=user_text,
            current_user_message_id=user_msg.id if user_msg else "",
            namespace=ns,
        )
        retrieval_ms = round((time.perf_counter() - t0) * 1000.0, 2)
        evidence = evidence.model_copy(update={"retrieval_ms": retrieval_ms})

        augmented = self.inject_memory_into_messages(
            messages,
            evidence.narrated_context,
            now_iso=ts,
            timezone_name=self.timezone_name,
            locale=memory_locale(self.locale, user_text),
        )
        return augmented, evidence, user_msg, recent_before, ts

    def finalize_gateway_turn(
        self,
        *,
        user_msg: MessageRecord | None,
        assistant_text: str,
        evidence: TurnEvidence | None,
        recent_before: list[MessageRecord],
        namespace: str = "default",
        session_id: str = "airi-default",
        memory_mode: str = "auto",
        now_iso: str | None = None,
        ttft_ms: float | None = None,
        total_ms: float | None = None,
    ) -> None:
        """Persist the assistant reply and its evidence independently of extraction."""
        if ttft_ms is not None:
            self._ttft_history_ms = [*self._ttft_history_ms[-199:], ttft_ms]
        if total_ms is not None:
            self._total_history_ms = [*self._total_history_ms[-199:], total_ms]

        if (memory_mode or "auto").strip().lower() != "auto" or user_msg is None:
            return

        ns = normalize_namespace(namespace)
        ts = now_iso or self._now_iso()
        if assistant_text.strip():
            assistant_msg = self.store.add_message(
                session_id,
                "assistant",
                assistant_text.strip(),
                namespace=ns,
                timezone_name=self.timezone_name,
                created_at=ts,
                reply_to_message_id=user_msg.id,
                index_for_retrieval=False,
            )
            if evidence is not None:
                self.store.record_turn_evidences(
                    assistant_msg.id,
                    item_ids=evidence.used_item_ids,
                    source_message_ids=evidence.used_source_message_ids,
                    ambiguity_ids=evidence.ambiguity_ids,
                )
        self._maybe_schedule_auto_summary(session_id, namespace=ns)

    def _schedule_session_summary(self, session_id: str, namespace: str) -> None:
        ns = normalize_namespace(namespace)

        async def _runner() -> None:
            try:
                await self.summarize_session(session_id, namespace=ns)
            except Exception as exc:
                logger.warning("Auto session summarization failed for %s/%s: %s", ns, session_id, exc)

        task = asyncio.create_task(_runner())
        self._bg_tasks.add(task)
        task.add_done_callback(self._bg_tasks.discard)

    def _maybe_schedule_auto_summary(self, session_id: str, *, namespace: str = "default") -> None:
        """Automatically summarize sessions in background on session switch or every N turns (0ms TTFT cost)."""
        if self.auto_summarize_turns <= 0 or not callable(
            getattr(self.llm, "summarize_session_facts", None)
        ):
            return
        ns = normalize_namespace(namespace)
        prev_sid = self._last_active_session.get(ns)
        self._last_active_session[ns] = session_id

        # 1. If AIRI moved to a new session, summarize the completed previous session (>=2 user turns).
        if prev_sid and prev_sid != session_id:
            prev_count = self.store.count_session_user_messages(prev_sid, namespace=ns)
            existing_prev = self.store.get_session_summary(prev_sid, namespace=ns)
            if prev_count >= 2 and (
                existing_prev is None or len(existing_prev.source_message_ids) < prev_count
            ):
                self._schedule_session_summary(prev_sid, ns)

        # 2. If the current long session hits auto_summarize_turns cadence, refresh its summary incrementally.
        cur_count = self.store.count_session_user_messages(session_id, namespace=ns)
        if cur_count >= self.auto_summarize_turns and cur_count % self.auto_summarize_turns == 0:
            self._schedule_session_summary(session_id, ns)

    async def chat_stream(self, session_id: str, user_text: str, *, now_iso: str | None = None):
        """True SSE streaming turn: yields meta -> delta* -> done -> memory_synced."""
        ts = now_iso or self._now_iso()
        turn_start = time.perf_counter()
        recent_before = self.store.get_session_messages(session_id, limit=self.recent_turn_limit)

        user_msg = self.store.add_message(
            session_id,
            "user",
            user_text,
            timezone_name=self.timezone_name,
            created_at=ts,
        )
        extraction = self._schedule_extraction("default")
        t0 = time.perf_counter()
        evidence = self._assemble_working_memory(
            session_id=session_id,
            query=user_text,
            current_user_message_id=user_msg.id,
        )
        retrieval_ms = round((time.perf_counter() - t0) * 1000.0, 2)
        evidence = evidence.model_copy(update={"retrieval_ms": retrieval_ms})

        yield {
            "event": "meta",
            "data": {
                "session_id": session_id,
                "user_message_id": user_msg.id,
                "query_intent": evidence.query_intent,
                "retrieval_ms": retrieval_ms,
                "used_item_ids": list(evidence.used_item_ids),
                "used_source_message_ids": list(evidence.used_source_message_ids),
                "narrated_context": evidence.narrated_context,
            },
        }

        chunks: list[str] = []
        usage: dict[str, int] | None = None
        ttft_ms: float | None = None

        stream_fn = getattr(self.llm, "generate_reply_stream", None)
        if callable(stream_fn):
            async for piece in stream_fn(
                user_message=user_text,
                recent_messages=recent_before,
                narrated_context=evidence.narrated_context,
                now_iso=ts,
                timezone_name=self.timezone_name,
            ):
                if "delta" in piece and piece["delta"]:
                    if ttft_ms is None:
                        ttft_ms = round((time.perf_counter() - turn_start) * 1000.0, 1)
                    chunks.append(piece["delta"])
                    yield {"event": "delta", "data": {"text": piece["delta"]}}
                elif "usage" in piece:
                    usage = piece["usage"]
        else:
            reply_text, usage = await self.llm.generate_reply(
                user_message=user_text,
                recent_messages=recent_before,
                narrated_context=evidence.narrated_context,
                now_iso=ts,
                timezone_name=self.timezone_name,
            )
            ttft_ms = round((time.perf_counter() - turn_start) * 1000.0, 1)
            chunks.append(reply_text)
            yield {"event": "delta", "data": {"text": reply_text}}

        full_reply = "".join(chunks).strip()
        total_ms = round((time.perf_counter() - turn_start) * 1000.0, 1)
        if ttft_ms is None:
            ttft_ms = total_ms

        self._ttft_history_ms = [*self._ttft_history_ms[-199:], ttft_ms]
        self._total_history_ms = [*self._total_history_ms[-199:], total_ms]

        assistant_msg = self.store.add_message(
            session_id,
            "assistant",
            full_reply,
            timezone_name=self.timezone_name,
            created_at=self._now_iso() if now_iso is None else ts,
            reply_to_message_id=user_msg.id,
            index_for_retrieval=False,
        )
        self.store.record_turn_evidences(
            assistant_msg.id,
            item_ids=evidence.used_item_ids,
            source_message_ids=evidence.used_source_message_ids,
            ambiguity_ids=evidence.ambiguity_ids,
        )

        yield {
            "event": "done",
            "data": {
                "assistant_message_id": assistant_msg.id,
                "reply": full_reply,
                "ttft_ms": ttft_ms,
                "total_ms": total_ms,
                "retrieval_ms": retrieval_ms,
                "usage": usage,
                "latency": self.latency_stats(),
            },
        }

        # Clients may disconnect while the durable extraction is still running.
        await asyncio.shield(extraction)
        yield {
            "event": "memory_synced",
            "data": {
                "item_count": len(self.store.list_items()),
                "open_ambiguities": len(self.store.list_open_ambiguities()),
            },
        }

    async def forget(self, target: str, *, namespace: str = "default") -> ForgetReceipt:
        """Flush pending extractions, then atomically cascade-delete target and all derivations in namespace."""
        await self.flush()
        receipt = self.store.forget(target, namespace=namespace)
        self._publish_event(
            "memory_forgotten",
            namespace=receipt.namespace,
            payload={
                "query": receipt.query,
                "deleted_items": len(receipt.deleted_item_ids),
                "deleted_consolidations": len(receipt.deleted_consolidation_ids),
                "deleted_summaries": len(receipt.deleted_session_summary_ids),
                "cleared_core_blocks": list(receipt.cleared_core_blocks),
            },
        )
        return receipt

    # ------------------------------------------------------------------
    # Background Extraction, Auto-Consolidation & Safety Guards
    # ------------------------------------------------------------------

    def _schedule_extraction(self, namespace: str) -> asyncio.Task:
        namespace = normalize_namespace(namespace)
        task = self._extraction_tasks.get(namespace)
        if task is None:
            task = asyncio.create_task(self._drain_extractions(namespace))
            self._extraction_tasks[namespace] = task
            self._bg_tasks.add(task)
            task.add_done_callback(self._bg_tasks.discard)
        return task

    async def _drain_extractions(self, namespace: str) -> None:
        try:
            async with self._extract_lock:
                await self._consume_pending_extractions(namespace)
        finally:
            self._extraction_tasks.pop(namespace, None)

    async def retry_quarantined_extraction(
        self, message_id: str, *, namespace: str = "default"
    ) -> MessageRecord | None:
        ns = normalize_namespace(namespace)
        async with self._extract_lock:
            if not self.store.requeue_extraction(message_id, namespace=ns):
                return None
            await self._consume_pending_extractions(ns)
            return self.store.get_message(message_id)

    async def _consume_pending_extractions(self, namespace: str) -> bool:
        while pending := self.store.get_pending_messages(namespace=namespace, limit=1):
            message = pending[0]
            if self.enable_trivial_gate and is_trivial_backchannel(message.content):
                self.store.mark_message_extracted(message.id)
                self.skipped_trivial_extractions += 1
                continue
            recent = self.store.get_messages_before(message, limit=self.recent_turn_limit)
            if not await self._run_extraction(message, recent, message.created_at):
                # Leave the failed head pending for the next drain or startup recovery.
                return False
        return True

    def _schedule_consolidation(self, namespace: str) -> None:
        ns = normalize_namespace(namespace)

        async def _runner() -> None:
            try:
                await self._run_consolidation(namespace=ns)
            except Exception as exc:
                logger.warning("Auto sleep consolidation failed for %s: %s", ns, exc)

        task = asyncio.create_task(_runner())
        self._bg_tasks.add(task)
        task.add_done_callback(self._bg_tasks.discard)

    async def _run_extraction(
        self,
        user_msg: MessageRecord,
        recent_before: list[MessageRecord],
        now_iso: str,
    ) -> bool:
        ns = normalize_namespace(user_msg.namespace)
        should_auto_consolidate = False
        try:
            candidates = self._select_extraction_candidates(
                user_msg.content, namespace=ns
            )
            resp, usage = await self.llm.extract_changes(
                user_message=user_msg.content,
                recent_messages=recent_before,
                candidate_items=candidates,
                now_iso=now_iso,
                timezone_name=self.timezone_name,
            )
            self.last_extraction_usage = usage
            operations = self._validate_and_plan_operations(
                user_msg.content, resp.changes, candidates, namespace=ns
            )
            touched = self.store.commit_extraction(
                user_msg.id, operations, created_at=now_iso
            )
            if touched:
                self._publish_event(
                    "memory_extracted",
                    namespace=ns,
                    payload={
                        "message_id": user_msg.id,
                        "touched_item_ids": touched,
                        "operation_count": len(operations),
                    },
                )
            if any(op.get("kind") == "add_ambiguity" for op in operations):
                self._publish_event(
                    "ambiguity_detected",
                    namespace=ns,
                    payload={"message_id": user_msg.id},
                )
            if touched and self.auto_consolidate_every > 0:
                self._mutations_since_consolidation[ns] += len(touched)
                if self._mutations_since_consolidation[ns] >= self.auto_consolidate_every:
                    self._mutations_since_consolidation[ns] = 0
                    should_auto_consolidate = True
        except (InvalidExtraction, StaleExtraction) as exc:
            quarantined = self.store.record_extraction_failure(
                user_msg.id, stale=isinstance(exc, StaleExtraction)
            )
            if quarantined:
                logger.warning("Quarantined memory extraction for %s: %s", user_msg.id, exc)
                self.extraction_errors.append(str(exc))
            # Try again within the structural failure budget or advance after quarantine.
            return True
        except Exception as exc:
            logger.warning("Background memory extraction failed: %s", exc)
            self.extraction_errors.append(str(exc))
            return False

        if should_auto_consolidate and callable(getattr(self.llm, "consolidate_insights", None)):
            self._schedule_consolidation(ns)
        return True

    def _select_extraction_candidates(
        self, text: str, *, namespace: str = "default", limit: int = 24
    ) -> list[ItemRecord]:
        """Hybrid-recall top relevant historical items + active/paused items within a namespace."""
        ns = normalize_namespace(namespace)
        sem_ids = self.store.search_items_semantic(text, namespace=ns, limit=16)
        fts_ids = self.store.search_items_fts(text, namespace=ns, limit=16)
        fused = reciprocal_rank_fusion([sem_ids, fts_ids], limit=16)
        one_hop = self.store.expand_one_hop_items(fused, limit=6)

        selected: dict[str, ItemRecord] = {}
        for iid in [*fused, *one_hop]:
            item = self.store.get_item(iid)
            if item is not None and item.namespace == ns:
                selected[iid] = item

        for status in ("active", "paused"):
            for item in self.store.list_items(namespace=ns, status=status)[:12]:
                selected.setdefault(item.id, item)
                if len(selected) >= limit:
                    break
        return list(selected.values())[:limit]

    def _validate_and_plan_operations(
        self,
        user_text: str,
        changes: list[ExtractedChange],
        candidates: list[ItemRecord],
        *,
        namespace: str = "default",
    ) -> list[dict]:
        ns = normalize_namespace(namespace)
        by_id = {item.id: item for item in candidates}
        zh = memory_locale(self.locale, user_text) == "zh"
        # A vague reference cannot authorize updates to multiple existing items.
        mutating = [
            c
            for c in changes
            if c.action in {"pause", "cancel", "retract", "reschedule", "resume", "update"}
            and c.target_item_id in by_id
        ]
        if len(mutating) >= 2 and not ALL_PLURAL_CUE.search(user_text):
            explicitly_named = [
                c
                for c in mutating
                if by_id[c.target_item_id].title.lower() in user_text.lower()  # type: ignore[index]
            ]
            if len(explicitly_named) < len(mutating):
                cand_ids = list(
                    dict.fromkeys(c.target_item_id for c in mutating if c.target_item_id)
                )
                return [
                    {
                        "kind": "add_ambiguity",
                        "source_quote": _anchor_quote(mutating[0].source_quote, user_text),
                        "candidate_item_ids": cand_ids,
                        "note": (
                            "原话可能指向多个事项且未明确区分，暂不修改，待用户确认。"
                            if zh else "Several items could match. Ask which one before changing anything."
                        ),
                    }
                ]

        operations: list[dict] = []
        for change in changes:
            quote = _anchor_quote(change.source_quote, user_text)
            if change.action == "ambiguous":
                cand_ids = [cid for cid in change.candidate_item_ids if cid in by_id]
                if not cand_ids and change.target_item_id in by_id:
                    cand_ids = [change.target_item_id]  # type: ignore[list-item]
                operations.append(
                    {
                        "kind": "add_ambiguity",
                        "source_quote": quote,
                        "candidate_item_ids": cand_ids,
                        "note": (change.ambiguity_note or (
                            "指代不明确，需向用户确认具体事项。" if zh else "Ask which item the user means."
                        )).strip(),
                    }
                )
                continue

            if change.action == "create" or (
                change.target_item_id is None and change.action in {"update", "reschedule"}
            ):
                # Avoid creating duplicate items via exact title match OR semantic near-duplicate dedup.
                title = (change.title or change.summary or quote).strip()[:40]
                exact_dup = next(
                    (it for it in candidates if it.title.strip().lower() == title.lower()),
                    None,
                )
                if exact_dup is None and self.enable_dedup:
                    dedup_probe = f"{title} {(change.summary or '').strip()}".strip()
                    sem_dup = self.store.find_similar_item(
                        dedup_probe, namespace=ns, kind=change.kind, threshold=0.84
                    )
                    # Only merge when epistemic trust allows it (inferred_belief never overwrites confirmed fact).
                    if sem_dup is not None and not (
                        change.epistemic_type == "inferred_belief"
                        and sem_dup.epistemic_type != "inferred_belief"
                    ):
                        exact_dup = sem_dup
                if exact_dup is not None:
                    operations.append(
                        {
                            "kind": "add_version",
                            "item_id": exact_dup.id,
                            "title": exact_dup.title,
                            "status": ACTION_TO_STATUS.get(change.action, "active"),
                            "change_type": "update" if change.action == "create" else change.action,
                            "summary": (change.summary or quote).strip(),
                            "effective_time_text": change.effective_time_text,
                            "valid_from": change.valid_from,
                            "valid_to": change.valid_to,
                            "condition_text": change.condition_text,
                            "entities": change.entities,
                            "related_item_ids": [
                                rid for rid in change.related_item_ids if rid in by_id
                            ],
                            "source_quote": quote,
                        }
                    )
                else:
                    operations.append(
                        {
                            "kind": "create_item",
                            "title": title,
                            "item_kind": change.kind,
                            "status": ACTION_TO_STATUS.get(change.action, "active"),
                            "epistemic_type": change.epistemic_type,
                            "confidence": change.confidence,
                            "change_type": "create",
                            "summary": (change.summary or title).strip(),
                            "effective_time_text": change.effective_time_text,
                            "valid_from": change.valid_from,
                            "valid_to": change.valid_to,
                            "condition_text": change.condition_text,
                            "entities": change.entities,
                            "related_item_ids": [
                                rid for rid in change.related_item_ids if rid in by_id
                            ],
                            "source_quote": quote,
                        }
                    )
                continue

            # State mutation targeting an existing item.
            target = by_id.get(change.target_item_id or "")
            if target is None:
                # Could not verify target item; record ambiguity instead of inventing or mis-editing.
                if change.candidate_item_ids:
                    operations.append(
                        {
                            "kind": "add_ambiguity",
                            "source_quote": quote,
                            "candidate_item_ids": [
                                cid for cid in change.candidate_item_ids if cid in by_id
                            ],
                            "note": change.ambiguity_note or (
                                "未找到唯一匹配的事项，保留原话待确认。"
                                if zh else "No unique item matched. Keep the quote for clarification."
                            ),
                        }
                    )
                continue

            # An inferred belief may never overwrite
            # or retract a user-confirmed fact or shared commitment.
            if (
                change.epistemic_type == "inferred_belief"
                and target.epistemic_type != "inferred_belief"
            ):
                continue

            # On resume, if LLM didn't repeat the time from the last active version, preserve it.
            eff_time = change.effective_time_text
            summary = (change.summary or quote).strip()
            if change.action == "resume" and not eff_time:
                last_active = next(
                    (v for v in reversed(target.versions) if v.status == "active" and v.effective_time_text),
                    None,
                )
                if last_active is not None:
                    eff_time = last_active.effective_time_text

            operations.append(
                {
                    "kind": "add_version",
                    "item_id": target.id,
                    "title": target.title,
                    "status": ACTION_TO_STATUS[change.action],
                    "change_type": change.action,
                    "summary": summary,
                    "effective_time_text": eff_time,
                    "valid_from": change.valid_from,
                    "valid_to": change.valid_to,
                    "condition_text": change.condition_text,
                    "entities": change.entities,
                    "related_item_ids": [
                        rid for rid in change.related_item_ids if rid in by_id
                    ],
                    "source_quote": quote,
                }
            )
        return operations

    # ------------------------------------------------------------------
    # Working-memory retrieval and ranking
    # ------------------------------------------------------------------

    def _assemble_working_memory(
        self,
        *,
        session_id: str,
        query: str,
        current_user_message_id: str,
        namespace: str = "default",
        as_of: str | None = None,
    ) -> TurnEvidence:
        ns = normalize_namespace(namespace)
        intent = classify_query_intent(query)
        asks_overview = bool(PLAN_OVERVIEW_CUE.search(query))

        # Use a historical snapshot when the caller supplies `as_of`.
        if as_of:
            snap_items = self.store.get_items_as_of(as_of, namespace=ns, include_closed=True)
            active_snap = [it for it in snap_items if it.status == "active"][:8]
            paused_snap = [it for it in snap_items if it.status == "paused"][:6]
            closed_snap = [it for it in snap_items if it.status not in {"active", "paused"}][:6]
            narrated_snap = self._narrate_context(
                locale=memory_locale(self.locale, query),
                intent="history",
                active_items=active_snap,
                paused_items=paused_snap,
                closed_items=closed_snap,
                pending_msgs=[],
                drilled_messages=[],
                ambiguities=[],
                as_of=as_of,
            )
            return TurnEvidence(
                namespace=ns,
                query_intent="history",
                embedding_backend=self.embedder.name,
                used_item_ids=tuple(it.id for it in [*active_snap, *paused_snap, *closed_snap]),
                as_of=as_of,
                narrated_context=narrated_snap,
            )

        # 1. Short-term pending raw utterances (background extraction not finished yet).
        pending_msgs = self.store.get_pending_messages(
            namespace=ns, exclude_id=current_user_message_id
        )

        # 2. Cached multi-channel hybrid retrieval + Cue-Triggered Associative Activation:
        item_sem_scored = self.store.search_items_semantic_scored(query, namespace=ns, limit=20)
        item_sem = [iid for iid, _ in item_sem_scored]
        top_item_sem = item_sem_scored[0][1] if item_sem_scored else 0.0
        item_fts = self.store.search_items_fts(query, namespace=ns, limit=20)

        msg_sem_scored = self.store.search_messages_semantic_scored(query, namespace=ns, limit=20)
        msg_sem = [mid for mid, _ in msg_sem_scored if mid != current_user_message_id]
        non_self_msg_scores = [sc for mid, sc in msg_sem_scored if mid != current_user_message_id]
        top_msg_sem = non_self_msg_scores[0] if non_self_msg_scores else 0.0
        msg_fts = [
            mid
            for mid in self.store.search_messages_fts(query, namespace=ns, limit=20)
            if mid != current_user_message_id
        ]

        matched_entities = self.store.match_known_entities(query, namespace=ns)
        is_pref_advice = bool(PREFERENCE_ADVICE_CUE.search(query)) or bool(
            ABSTRACT_INFERENCE_CUE.search(query)
        )
        # Fix declarative statement blindspot: wake up associative memory when a non-question
        # statement mentions known entities or hits high-confidence semantic memory (>=0.52).
        cue_activated = (not is_trivial_backchannel(query)) and (
            bool(matched_entities) or top_item_sem >= 0.52 or top_msg_sem >= 0.52
        )
        effective_associative = (intent != "casual") or cue_activated or is_pref_advice

        fused_item_ids = reciprocal_rank_fusion([item_sem, item_fts], limit=12)
        graph_item_ids = self.store.expand_one_hop_items(fused_item_ids[:6], limit=6)
        fused_msg_ids = reciprocal_rank_fusion([msg_sem, msg_fts], limit=12)

        # Rank linked entities and evidence with personalized PageRank.
        ppr_item_ids: list[str] = []
        ppr_msg_scores: dict[str, float] = {}
        ppr_sum_ids: list[str] = []
        if self.enable_ppr and effective_associative:
            ppr_items, ppr_msgs, ppr_sums = self.store.run_heterogeneous_ppr(
                query=query,
                seed_item_ids=fused_item_ids[:8],
                seed_message_ids=fused_msg_ids[:8],
                namespace=ns,
                top_items=8,
                top_messages=12,
                top_summaries=4,
            )
            ppr_item_ids = [iid for iid, _ in ppr_items]
            ppr_msg_scores = {
                mid: sc for mid, sc in ppr_msgs if mid != current_user_message_id
            }
            ppr_sum_ids = [sid for sid, _ in ppr_sums]

        candidate_item_ids = reciprocal_rank_fusion(
            [fused_item_ids, graph_item_ids, ppr_item_ids], limit=14
        )

        # Hydrate candidate items strictly within this namespace.
        all_items = self.store.list_items(namespace=ns)
        by_item_id = {it.id: it for it in all_items}
        recalled_items: list[ItemRecord] = []
        for iid in candidate_item_ids:
            item = by_item_id.get(iid)
            if item is not None:
                recalled_items.append(item)

        recalled_ids_set = {it.id for it in recalled_items}
        for it in all_items:
            if it.id not in recalled_ids_set and _item_directly_matched(query, it):
                recalled_items.insert(0, it)
                recalled_ids_set.add(it.id)

        # Weight recalled items by recency and use while keeping direct matches first.
        base_rank = {it.id: idx for idx, it in enumerate(recalled_items)}
        recalled_items.sort(
            key=lambda it: (
                0 if _item_directly_matched(query, it) else 1,
                base_rank[it.id]
                - 0.6
                * compute_actr_activation(
                    it.access_count, it.salience, it.last_mentioned_at or it.updated_at
                ),
            )
        )

        # 3. Companion Mention Gating:
        active_to_show: list[ItemRecord] = []
        paused_to_show: list[ItemRecord] = []
        closed_to_show: list[ItemRecord] = []
        top_candidate_gate = set(candidate_item_ids[:6])

        if asks_overview:
            active_to_show = [it for it in all_items if it.status == "active"][:8]
            paused_to_show = [it for it in all_items if it.status == "paused"][:6]
        else:
            for item in recalled_items:
                direct = _item_directly_matched(query, item)
                if item.status == "active":
                    if direct or (effective_associative and item.id in top_candidate_gate):
                        active_to_show.append(item)
                elif item.status == "paused":
                    # Strict anti-nag rule: NEVER surface paused plans unless directly referenced!
                    if direct:
                        paused_to_show.append(item)
                else:
                    if direct or intent in {"history", "both"}:
                        closed_to_show.append(item)

        active_to_show = active_to_show[:6]
        paused_to_show = paused_to_show[:4]
        closed_to_show = closed_to_show[:4]
        selected_items = [*active_to_show, *paused_to_show, *closed_to_show]

        # 4. Raw provenance drill-down + Multi-Feature Reranking:
        drilled_messages: list[MessageRecord] = []
        seen_msg_ids: set[str] = {current_user_message_id}
        if effective_associative or selected_items:
            for item in selected_items:
                versions_to_cite = (
                    item.versions if intent in {"history", "both"} else item.versions[-2:]
                )
                for ver in versions_to_cite:
                    if ver.source_message_id not in seen_msg_ids:
                        msg = self.store.get_message(ver.source_message_id)
                        if msg is not None and msg.namespace == ns:
                            drilled_messages.append(msg)
                            seen_msg_ids.add(msg.id)

            if effective_associative:
                raw_pool_ids = list(
                    dict.fromkeys([*fused_msg_ids, *ppr_msg_scores.keys()])
                )[:20]
                if self.enable_reranker and raw_pool_ids:
                    sem_r_map = {mid: r for r, mid in enumerate(msg_sem, start=1)}
                    fts_r_map = {mid: r for r, mid in enumerate(msg_fts, start=1)}
                    cand_dicts = []
                    for mid in raw_pool_ids:
                        mrec = self.store.get_message(mid)
                        if mrec is not None and mrec.namespace == ns:
                            cand_dicts.append(
                                {
                                    "id": mrec.id,
                                    "session_id": mrec.session_id,
                                    "content": mrec.content,
                                    "created_at": mrec.created_at,
                                    "sem_rank": sem_r_map.get(mrec.id),
                                    "fts_rank": fts_r_map.get(mrec.id),
                                    "ppr_score": ppr_msg_scores.get(mrec.id, 0.0),
                                }
                            )
                    ordered_raw_ids = self.reranker.rerank_messages(
                        query,
                        cand_dicts,
                        intent=intent if intent != "casual" else "current",
                        limit=8,
                        diverse_sessions=bool(MULTI_SESSION_CUE.search(query)),
                    )
                else:
                    ordered_raw_ids = raw_pool_ids[:6]

                for mid in ordered_raw_ids[:5]:
                    if mid in seen_msg_ids:
                        continue
                    msg = self.store.get_message(mid)
                    if msg is None or msg.namespace != ns:
                        continue
                    if self._belongs_to_hidden_paused_item(msg.id, paused_to_show, all_items):
                        continue
                    drilled_messages.append(msg)
                    seen_msg_ids.add(msg.id)
                    if intent in {"current", "both"} or cue_activated:
                        for nb in self.store.get_neighbor_user_messages(msg.id, radius=1):
                            if nb.id not in seen_msg_ids and not self._belongs_to_hidden_paused_item(
                                nb.id, paused_to_show, all_items
                            ):
                                drilled_messages.append(nb)
                                seen_msg_ids.add(nb.id)
                    if len(drilled_messages) >= 10:
                        break

        # 5. Open ambiguities in this namespace.
        open_ambiguities = self.store.list_open_ambiguities(namespace=ns, limit=4)
        relevant_ambiguities: list[AmbiguityRecord] = []
        for amb in open_ambiguities:
            amb_items = [by_item_id.get(cid) for cid in amb.candidate_item_ids]
            if intent != "casual" or any(
                it is not None and _item_directly_matched(query, it) for it in amb_items
            ):
                relevant_ambiguities.append(amb)

        # 5b. Consolidated insights (strictly withholding hidden paused items).
        relevant_consolidations: list[ConsolidationRecord] = []
        if effective_associative:
            hidden_paused_ids = {
                it.id
                for it in all_items
                if it.status == "paused" and it.id not in {p.id for p in paused_to_show}
            }
            sem_cons = self.store.search_consolidations(
                query, namespace=ns, limit=3, include_top_if_empty=is_pref_advice
            )
            ent_cons = self.store.search_consolidations_by_entities(
                query, namespace=ns, limit=4
            )
            seen_cids: set[str] = set()
            for cons in [*ent_cons, *sem_cons]:
                if cons.id in seen_cids:
                    continue
                if cons.source_item_ids and any(
                    iid in hidden_paused_ids for iid in cons.source_item_ids
                ):
                    continue
                relevant_consolidations.append(cons)
                seen_cids.add(cons.id)
                if len(relevant_consolidations) >= 4:
                    break

        # Retrieve session summaries by resolved entities and graph relevance.
        relevant_summaries: list[SessionSummaryRecord] = []
        if effective_associative:
            sem_ss_live = self.store.search_session_summaries(query, namespace=ns, limit=3)
            ent_ss_live = self.store.search_session_summaries_by_entities(
                query, namespace=ns, limit=4
            )
            all_ss_ns = self.store.list_session_summaries(namespace=ns) if ppr_sum_ids else []
            ss_by_id = {s.id: s for s in all_ss_ns}
            ppr_ss_live = [ss_by_id[sid] for sid in ppr_sum_ids if sid in ss_by_id]
            seen_ss_live: set[str] = set()
            live_pool = [*ent_ss_live, *sem_ss_live, *ppr_ss_live]
            if TEMPORAL_ORIGIN_CUE.search(query):
                earliest_live = sorted(
                    ent_ss_live or self.store.list_session_summaries(namespace=ns),
                    key=lambda s: _chrono_sort_key(s.session_date, s.session_id),
                )[:2]
                live_pool = [*earliest_live, *live_pool]
            for ss in live_pool:
                if ss.id not in seen_ss_live:
                    relevant_summaries.append(ss)
                    seen_ss_live.add(ss.id)
                if len(relevant_summaries) >= 4:
                    break

        # Include core blocks whose sources are relevant to the current turn.
        hidden_paused_ids_all = {
            it.id
            for it in all_items
            if it.status == "paused" and it.id not in {p.id for p in paused_to_show}
        }
        core_blocks = [
            blk
            for blk in self.store.get_core_memory_blocks(namespace=ns)
            if not (
                blk.source_item_ids
                and any(iid in hidden_paused_ids_all for iid in blk.source_item_ids)
            )
        ]

        # 6. Narrate working memory cleanly (with deterministic relative-time anchoring).
        narrated = self._narrate_context(
            locale=memory_locale(self.locale, query),
            intent=intent,
            active_items=active_to_show,
            paused_items=paused_to_show,
            closed_items=closed_to_show,
            pending_msgs=pending_msgs,
            drilled_messages=drilled_messages,
            ambiguities=relevant_ambiguities,
            consolidations=relevant_consolidations,
            session_summaries=relevant_summaries,
            core_blocks=core_blocks,
            show_excerpts=effective_associative,
        )

        used_item_ids = tuple(dict.fromkeys(it.id for it in selected_items))
        used_source_ids = tuple(
            dict.fromkeys(
                [
                    *(m.id for m in drilled_messages),
                    *(m.id for m in pending_msgs),
                    *(v.source_message_id for it in selected_items for v in it.versions),
                ]
            )
        )
        return TurnEvidence(
            namespace=ns,
            query_intent=intent,  # type: ignore[arg-type]
            embedding_backend=self.embedder.name,
            used_item_ids=used_item_ids,
            used_source_message_ids=used_source_ids,
            used_consolidation_ids=tuple(c.id for c in relevant_consolidations),
            used_session_summary_ids=tuple(s.id for s in relevant_summaries),
            pending_message_ids=tuple(m.id for m in pending_msgs),
            ambiguity_ids=tuple(a.id for a in relevant_ambiguities),
            cue_activated=cue_activated,
            narrated_context=narrated,
        )

    @staticmethod
    def _belongs_to_hidden_paused_item(
        message_id: str,
        shown_paused: list[ItemRecord],
        all_items: list[ItemRecord],
    ) -> bool:
        shown_ids = {it.id for it in shown_paused}
        for item in all_items:
            if item.status == "paused" and item.id not in shown_ids:
                if any(v.source_message_id == message_id for v in item.versions):
                    return True
        return False

    def _narrate_context(
        self,
        *,
        locale: str = "en",
        intent: str,
        active_items: list[ItemRecord],
        paused_items: list[ItemRecord],
        closed_items: list[ItemRecord],
        pending_msgs: list[MessageRecord],
        drilled_messages: list[MessageRecord],
        ambiguities: list[AmbiguityRecord],
        consolidations: list[ConsolidationRecord] = (),  # type: ignore[assignment]
        session_summaries: list[SessionSummaryRecord] = (),  # type: ignore[assignment]
        core_blocks: list[CoreMemoryBlock] = (),  # type: ignore[assignment]
        show_excerpts: bool = False,
        as_of: str | None = None,
    ) -> str:
        zh = locale == "zh"

        def localize(en: str, cn: str) -> str:
            return cn if zh else en

        def timeline(item: ItemRecord) -> str:
            versions = " -> ".join(_format_version_line(version, zh=zh) for version in item.versions)
            return localize("  History: ", "  演变记录：") + versions

        condition_label = localize(" | Condition: ", "｜条件：")
        sections: list[str] = []
        if as_of:
            sections.append(localize(
                f"[Memory state as of {as_of}]", f"[历史时间截面快照（截至 {as_of}）]"
            ))

        if core_blocks:
            lines = [localize("[Core memory]", "[常驻核心记忆（始终生效的长期默契与边界）]")]
            lines.extend(f"- {blk.block_key}: {blk.content}" for blk in core_blocks[:4])
            sections.append("\n".join(lines))

        if pending_msgs:
            lines = [localize(
                "[Recent user statements awaiting extraction. Prefer these over earlier state.]",
                "[刚说的最新原话（后台尚未整理完，优先级高于下方旧状态）]",
            )]
            lines.extend(f"- ({msg.created_at}) {msg.content}" for msg in pending_msgs[-4:])
            sections.append("\n".join(lines))

        if consolidations:
            lines = [localize("[Long-term understanding and preferences]", "[对你的长期了解与偏好（基于历史归纳）]")]
            lines.extend(f"- {cons.topic}: {cons.summary}" for cons in consolidations)
            sections.append("\n".join(lines))

        if session_summaries:
            lines = [localize("[Relevant session summaries]", "[相关过往会话消歧摘要]")]
            for ss in session_summaries[:3]:
                facts_join = localize("; ", "；").join(ss.facts[:3])
                detail = localize(f" (Key facts: {facts_join})", f"（要点：{facts_join}）") if facts_join else ""
                lines.append(f"- ({ss.session_date}) {ss.summary}{detail}")
            sections.append("\n".join(lines))

        if active_items:
            lines = [localize("[Current active items]", "[当前仍在进行的事项（最新有效状态）]")]
            for item in active_items:
                cur = item.current
                desc = cur.summary if cur else item.title
                time_part = localize(" | Effective time: ", "｜生效时间：") + cur.effective_time_text if cur and cur.effective_time_text else ""
                condition = condition_label + cur.condition_text if cur and cur.condition_text else ""
                lines.append(f"- {item.title}: {desc}{time_part}{condition}")
                if intent in {"history", "both"} and len(item.versions) > 1:
                    lines.append(timeline(item))
                elif len(item.versions) > 1:
                    prev = item.versions[-2]
                    lines.append(localize(
                        f'  The earlier version "{prev.summary}" has been superseded.',
                        f'  旧版本“{prev.summary}”已被当前状态取代。',
                    ))
            sections.append("\n".join(lines))

        if paused_items:
            lines = [localize(
                "[Paused items. Let them rest unless the user asks or resumes.]",
                "[已暂停/搁置的事项（当前不进行，除非用户主动问起或恢复，不要催促）]",
            )]
            paused = localize("paused", "已暂停")
            for item in paused_items:
                cur = item.current
                lines.append(f"- {item.title} ({paused}): {cur.summary if cur else item.title}")
                if len(item.versions) > 1:
                    lines.append(timeline(item))
            sections.append("\n".join(lines))

        if closed_items:
            lines = [localize("[Closed or retracted items]", "[已结束/已撤回的历史事项]")]
            for item in closed_items:
                cur = item.current
                condition = condition_label + cur.condition_text if cur and cur.condition_text else ""
                lines.append(f"- {item.title} ({item.status}): {cur.summary if cur else item.title}{condition}")
                if len(item.versions) > 1:
                    lines.append(timeline(item))
            sections.append("\n".join(lines))

        if ambiguities:
            lines = [localize(
                "[Unresolved references. Ask the user which item they meant.]",
                "[未决指代疑点（以下话指代不明，暂未修改事项，请自然向用户确认）]",
            )]
            for amb in ambiguities:
                titles = []
                for cid in amb.candidate_item_ids:
                    item = self.store.get_item(cid)
                    if item is not None:
                        titles.append(f"{item.title} ({item.status})")
                candidates = " / ".join(titles) or localize("multiple items", "多个已有事项")
                lines.append(localize(
                    f'- Quote: "{amb.source_quote}" | Possible matches: {candidates} | {amb.note}',
                    f"- 原话：“{amb.source_quote}”｜可能指向：{candidates}｜疑点：{amb.note}",
                ))
            sections.append("\n".join(lines))

        if drilled_messages and (show_excerpts or intent in {"history", "both", "current"}):
            lines = [localize("[Relevant verbatim user excerpts]", "[可核对的相关用户原话]")]
            for msg in drilled_messages[:6]:
                anchor_hint = anchor_relative_dates(msg.content, msg.created_at)
                quoted = localize(f'"{msg.content}"', f"“{msg.content}”")
                lines.append(f"- ({msg.created_at}){anchor_hint} {quoted}")
            sections.append("\n".join(lines))

        if not sections and intent in {"history", "both", "current"}:
            sections.append(localize(
                "[No relevant memory found. Acknowledge the gap and avoid inventing past details.]",
                "[记忆检索结果：未找到与本轮问题相关的记录。请如实说不记得，避免编造旧事。]",
            ))

        return "\n\n".join(sections)
