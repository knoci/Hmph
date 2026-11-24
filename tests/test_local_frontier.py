"""Unit tests for v2.5 frontier memory primitives:
bipartite PPR, semantic dedup, epistemic protection, relative-time anchoring, and MCP server.
"""

from __future__ import annotations

import pytest

from hmph.local_memory import (
    ExtractedChange,
    ExtractionResponse,
    HashingEmbedder,
    LocalMemoryEngine,
)
from hmph.local_memory.mcp_server import LocalMCPServer
from hmph.local_memory.reranker import (
    FeatureReranker,
    anchor_relative_dates,
    extract_anchored_search_tokens,
)


class FrontierMockLLM:
    model_name = "frontier-mock"

    async def extract_changes(
        self,
        *,
        user_message,
        recent_messages,
        candidate_items,
        now_iso,
        timezone_name,
    ):
        if "博物馆" in user_message or "Museum" in user_message:
            return (
                ExtractionResponse(
                    changes=[
                        ExtractedChange(
                            action="create",
                            title=user_message[:24],
                            kind="episode",
                            summary=user_message,
                            entities=["museum", "exhibition"],
                            source_quote=user_message,
                        )
                    ]
                ),
                {"total_tokens": 30},
            )
        if "周六练琴" in user_message:
            return (
                ExtractionResponse(
                    changes=[
                        ExtractedChange(
                            action="create",
                            title="练琴计划",
                            kind="plan",
                            summary="每周六练琴",
                            effective_time_text="周六",
                            entities=["练琴"],
                            epistemic_type="shared_commitment",
                            source_quote=user_message,
                        )
                    ]
                ),
                {"total_tokens": 30},
            )
        if "每周六练习钢琴" in user_message:
            # Near-duplicate create should merge via semantic dedup.
            return (
                ExtractionResponse(
                    changes=[
                        ExtractedChange(
                            action="create",
                            title="每周六练习钢琴",
                            kind="plan",
                            summary="每周六练琴",
                            entities=["练琴"],
                            source_quote=user_message,
                        )
                    ]
                ),
                {"total_tokens": 30},
            )
        if "可能不想练琴" in user_message:
            piano = next((it for it in candidate_items if "练琴" in it.title), None)
            if piano:
                return (
                    ExtractionResponse(
                        changes=[
                            ExtractedChange(
                                action="cancel",
                                target_item_id=piano.id,
                                summary="推断用户可能不想练琴",
                                epistemic_type="inferred_belief",
                                confidence=0.4,
                                source_quote=user_message,
                            )
                        ]
                    ),
                    {"total_tokens": 30},
                )
        return ExtractionResponse(changes=[]), {"total_tokens": 10}

    async def generate_reply(self, **kwargs):
        return "ok", {"total_tokens": 10}

    async def answer_from_context(self, **kwargs):
        return "ok", {"total_tokens": 10}

    async def consolidate_insights(self, *, items, existing_consolidations, timezone_name):
        if not items:
            return [], {"total_tokens": 10}
        return (
            [
                {
                    "topic": "日常音乐习惯",
                    "summary": "用户坚持在周末安排练琴，偏好稳定的每周节奏。",
                    "source_item_ids": [items[0].id],
                    "confidence": 0.92,
                }
            ],
            {"total_tokens": 45},
        )


@pytest.mark.asyncio
async def test_bipartite_ppr_recalls_cross_session_entity_cluster(tmp_path):
    engine = LocalMemoryEngine(tmp_path / "ppr.db", llm=FrontierMockLLM(), embedder=HashingEmbedder())
    museums = [
        ("s1", "Visited Science Museum for space exhibition"),
        ("s2", "Attended lecture at Museum of Contemporary Art"),
        ("s3", "Saw ancient pottery at Metropolitan Museum of Art"),
        ("s4", "Walked through Museum of History"),
        ("s5", "Explored Modern Art Museum"),
        ("s6", "Took photos at Natural History Museum"),
    ]
    for sid, text in museums:
        await engine.ingest_user_message(sid, text, await_extraction=True)

    ppr_items, ppr_msgs = engine.store.run_bipartite_ppr(
        query="What is the order of the six museums I visited?",
        seed_item_ids=[],
        seed_message_ids=[],
        top_items=10,
        top_messages=10,
    )
    assert len(ppr_items) == 6
    assert len(ppr_msgs) == 6
    await engine.close()


@pytest.mark.asyncio
async def test_semantic_dedup_and_epistemic_protection(tmp_path):
    engine = LocalMemoryEngine(
        tmp_path / "epistemic.db",
        llm=FrontierMockLLM(),
        embedder=HashingEmbedder(),
        enable_dedup=True,
    )
    await engine.ingest_user_message("s1", "约好每周六练琴", await_extraction=True)
    assert len(engine.list_items()) == 1
    assert engine.list_items()[0].epistemic_type == "shared_commitment"

    # Low-confidence inferred_belief MUST NOT cancel a user-confirmed commitment.
    await engine.ingest_user_message("s2", "最近有点累可能不想练琴", await_extraction=True)
    piano = engine.list_items()[0]
    assert piano.status == "active"
    assert len(piano.versions) == 1
    await engine.close()


def test_relative_time_anchoring_zh_and_en():
    en_hint = anchor_relative_dates(
        "I went to the gallery yesterday, started classes two weeks ago, and adopted three dogs three years ago.",
        "2023/05/10 (Wed) 14:00",
    )
    assert "2023-05-09" in en_hint
    assert "2023-04-26" in en_hint
    assert "2020" in en_hint

    toks = extract_anchored_search_tokens("I adopted my first three dogs three years ago.", "2023-06-14T15:00:00")
    assert "2020" in toks

    zh_hint = anchor_relative_dates("昨天去了展览，上周报了法语班，三年前养了猫", "2026-11-21T16:00:00+08:00")
    assert "2026-11-20" in zh_hint
    assert "2026-11-14" in zh_hint
    assert "2023" in zh_hint


def test_session_burst_aware_reranker_keeps_two_same_session_museums():
    reranker = FeatureReranker()
    cands = [
        {"id": "m1", "session_id": "s1", "content": "Visited Science Museum today", "created_at": "2023/01/15 10:00", "sem_rank": 1, "fts_rank": 1, "ppr_score": 0.05},
        {"id": "m2", "session_id": "s1", "content": "Also visited Museum of Contemporary Art today", "created_at": "2023/01/15 15:00", "sem_rank": 2, "fts_rank": 2, "ppr_score": 0.05},
        {"id": "m3", "session_id": "s2", "content": "Went to Metropolitan Museum of Art", "created_at": "2023/02/01 10:00", "sem_rank": 3, "fts_rank": 3, "ppr_score": 0.04},
        {"id": "m4", "session_id": "s3", "content": "Visited Museum of History", "created_at": "2023/02/10 10:00", "sem_rank": 4, "fts_rank": 4, "ppr_score": 0.04},
        {"id": "m5", "session_id": "s4", "content": "Went to Modern Art Museum", "created_at": "2023/03/01 10:00", "sem_rank": 5, "fts_rank": 5, "ppr_score": 0.04},
        {"id": "m6", "session_id": "s5", "content": "Visited Natural History Museum", "created_at": "2023/03/20 10:00", "sem_rank": 6, "fts_rank": 6, "ppr_score": 0.04},
        {"id": "noise", "session_id": "s6", "content": "Bought some groceries and bread", "created_at": "2023/03/25 10:00", "sem_rank": 20, "fts_rank": None, "ppr_score": 0.0},
    ]
    top = reranker.rerank_messages(
        "What is the chronological order of the six museums I visited?",
        cands,
        intent="history",
        limit=6,
        diverse_sessions=True,
    )
    assert {"m1", "m2", "m3", "m4", "m5", "m6"}.issubset(set(top))


@pytest.mark.asyncio
async def test_anchored_fts_year_search_and_consolidation_zero_resurrection(tmp_path):
    engine = LocalMemoryEngine(tmp_path / "v3.db", llm=FrontierMockLLM(), embedder=HashingEmbedder())
    # 1. Anchored year token indexed into FTS5 without modifying raw text.
    msg = engine.store.add_message(
        "s_dogs",
        "user",
        "Audrey: I adopted my first three dogs three years ago from a rescue shelter.",
        created_at="2023-06-15T10:00:00",
    )
    hits = engine.store.search_messages_fts("2020 adopted dogs", limit=5)
    assert msg.id in hits
    assert "2020" not in (engine.store.get_message(msg.id).content)
    assert engine.rebuild_fts_anchors() >= 1
    assert msg.id in engine.store.search_messages_fts("2020", limit=5)

    # 2. Sleep consolidation + zero-resurrection cascade forget.
    await engine.ingest_user_message("s_piano", "约好每周六练琴", await_extraction=True)
    report = await engine.consolidate()
    assert len(report.consolidations) == 1
    assert report.consolidations[0].topic == "日常音乐习惯"
    assert len(report.consolidations[0].source_quotes) == 1

    # Recall should surface consolidated insight in narrated context.
    ev = engine.recall_context("有什么音乐练习建议？")
    assert "日常音乐习惯" in ev.narrated_context

    # Cascade forget MUST delete the derived consolidation too (zero resurrection!).
    receipt = await engine.forget("练琴")
    assert len(receipt.deleted_item_ids) == 1
    assert len(receipt.deleted_consolidation_ids) == 1
    assert engine.list_consolidations() == []
    await engine.close()


@pytest.mark.asyncio
async def test_mcp_server_seven_tools_roundtrip(tmp_path):
    engine = LocalMemoryEngine(tmp_path / "mcp.db", llm=FrontierMockLLM(), embedder=HashingEmbedder())
    mcp = LocalMCPServer(engine)

    init = await mcp.handle_request({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}})
    assert init is not None and init["result"]["serverInfo"]["name"] == "hmph-local-memory"

    tools = await mcp.handle_request({"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}})
    tool_names = {t["name"] for t in tools["result"]["tools"]}
    assert tool_names == {
        "hmph_remember",
        "hmph_recall",
        "hmph_list_memories",
        "hmph_forget",
        "hmph_consolidate",
        "hmph_graph",
        "hmph_toggle_pause",
        "hmph_core_memory",
    }

    rem = await mcp.handle_request(
        {
            "jsonrpc": "2.0",
            "id": 3,
            "method": "tools/call",
            "params": {"name": "hmph_remember", "arguments": {"text": "约好每周六练琴"}},
        }
    )
    assert not rem["result"].get("isError")

    cons = await mcp.handle_request(
        {
            "jsonrpc": "2.0",
            "id": 30,
            "method": "tools/call",
            "params": {"name": "hmph_consolidate", "arguments": {}},
        }
    )
    assert not cons["result"].get("isError")
    assert "日常音乐习惯" in cons["result"]["content"][0]["text"]

    rec = await mcp.handle_request(
        {
            "jsonrpc": "2.0",
            "id": 4,
            "method": "tools/call",
            "params": {"name": "hmph_recall", "arguments": {"query": "练琴约在周几？"}},
        }
    )
    assert "练琴" in rec["result"]["content"][0]["text"]

    fg = await mcp.handle_request(
        {
            "jsonrpc": "2.0",
            "id": 5,
            "method": "tools/call",
            "params": {"name": "hmph_forget", "arguments": {"target": "练琴"}},
        }
    )
    assert not fg["result"].get("isError")
    assert engine.list_items() == []
    assert engine.list_consolidations() == []
    await engine.close()


@pytest.mark.asyncio
async def test_session_summaries_core_memory_proactive_and_zero_resurrection(tmp_path):
    class V5MockLLM(FrontierMockLLM):
        async def summarize_session_facts(self, *, session_id, session_date, turns, timezone_name):
            return (
                {
                    "summary": "Audrey adopted her first three dogs Pepper, Precious, and Panda in 2020.",
                    "facts": [
                        "On 2023-03-27, Audrey stated she has had her three dogs (Pepper, Precious, Panda) for 3 years (adopted ~2020)."
                    ],
                    "entities": ["Audrey", "Pepper", "Precious", "Panda", "dogs"],
                },
                {"total_tokens": 50},
            )

    engine = LocalMemoryEngine(
        tmp_path / "v5.db", llm=V5MockLLM(), embedder=HashingEmbedder(), auto_consolidate_every=0
    )
    # 1. Raw message has pronoun 'them' without the word 'dogs', but session_summary resolves coreference!
    msg = engine.store.add_message(
        "s1",
        "user",
        "Audrey: I've had them for 3 years! Their names are Pepper, Precious and Panda.",
        created_at="2023-03-27T13:10:00+00:00",
    )
    bf = await engine.backfill_session_summaries()
    assert bf["summarized"] == 1
    ss_hits = engine.store.search_session_summaries("Which year did Audrey adopt her dogs?")
    assert len(ss_hits) == 1
    assert "2020" in ss_hits[0].facts[0]

    # 2. Active plan + proactive boundary check (paused plan MUST NEVER initiate!).
    await engine.ingest_user_message("s2", "约好每周六练琴", await_extraction=True)
    pro_active = engine.get_proactive_suggestion(cooldown_hours=0.0)
    assert pro_active.should_initiate is True
    assert pro_active.reason == "active_plan_checkin"

    piano = engine.list_items()[0]
    await engine.toggle_item_pause(piano.id, session_id="s2")
    pro_paused = engine.get_proactive_suggestion(cooldown_hours=0.0)
    assert pro_paused.should_initiate is False
    assert pro_paused.excluded_paused_count == 1

    # 3. Core memory block + cascade forget zero-resurrection across summaries & core blocks.
    engine.set_core_memory_block(
        "user_profile",
        "喜欢每周六练琴",
        source_item_ids=[piano.id],
    )
    fg_piano = await engine.forget("练琴")
    assert "user_profile" in fg_piano.cleared_core_blocks
    assert engine.get_core_memory_blocks() == []

    fg_msg = await engine.forget(msg.id)
    assert len(fg_msg.deleted_session_summary_ids) == 1
    assert engine.list_session_summaries() == []
    await engine.close()


@pytest.mark.asyncio
async def test_entity_scoped_recall_and_auto_session_summarizer(tmp_path):
    class AutoSumMockLLM(FrontierMockLLM):
        def __init__(self) -> None:
            self.summarized_sessions: list[str] = []

        async def summarize_session_facts(self, *, session_id, session_date, turns, timezone_name):
            self.summarized_sessions.append(session_id)
            return (
                {
                    "summary": f"Summary for {session_id}: Andrew and his girlfriend did an activity.",
                    "facts": [f"In {session_id}, Andrew and his girlfriend enjoyed volunteering/boardgames."],
                    "entities": ["Andrew", "girlfriend"],
                },
                {"total_tokens": 30},
            )

    llm = AutoSumMockLLM()
    engine = LocalMemoryEngine(
        tmp_path / "v6.db",
        llm=llm,
        embedder=HashingEmbedder(),
        auto_consolidate_every=0,
        auto_summarize_turns=6,
    )
    # 1. Seed 3 summaries for Andrew + girlfriend that do NOT contain the word "indoor".
    for idx, desc in enumerate(
        (
            "Andrew and his girlfriend volunteered at a pet shelter.",
            "Andrew and his girlfriend attended a wine tasting.",
            "Andrew and his girlfriend played board games with Toby.",
        ),
        start=1,
    ):
        engine.store.upsert_session_summary(
            f"hist_{idx}",
            session_date=f"2023-0{idx}-15",
            summary=desc,
            facts=[desc],
            entities=["Andrew", "girlfriend"],
            source_message_ids=[],
        )

    ent_hits = engine.store.search_session_summaries_by_entities(
        "What indoor activities has Andrew pursued with his girlfriend?", limit=6
    )
    assert len(ent_hits) == 3

    # 2. Live gateway session switch auto-summarizes completed session (>=2 user turns) in background!
    for turn in ("今天我跟对象去了新开的猫咖。", "对了，我们周末还想去爬山。"):
        aug, ev, u_msg, rec, ts = engine.prepare_gateway_turn(
            [{"role": "user", "content": turn}],
            session_id="day1",
        )
        engine.finalize_gateway_turn(
            user_msg=u_msg,
            assistant_text="好的记下啦",
            evidence=ev,
            recent_before=rec,
            session_id="day1",
        )
    # Switch to day2 -> triggers automatic background summarization of day1!
    aug2, ev2, u_msg2, rec2, ts2 = engine.prepare_gateway_turn(
        [{"role": "user", "content": "早安！"}],
        session_id="day2",
    )
    engine.finalize_gateway_turn(
        user_msg=u_msg2,
        assistant_text="早安！",
        evidence=ev2,
        recent_before=rec2,
        session_id="day2",
    )
    await engine.flush()
    assert "day1" in llm.summarized_sessions
    assert engine.store.get_session_summary("day1") is not None
    await engine.close()
