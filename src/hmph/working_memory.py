"""Bounded evidence assembly. Slot coverage describes retrieved support, not truth."""

import re
from dataclasses import dataclass
from datetime import datetime
from typing import Literal
from uuid import UUID
from zoneinfo import ZoneInfo

from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import case, or_, select

from hmph.lexical import bm25_order, tokenize
from hmph.memory import FUNCTIONAL, HINTS, PAST, Memory
from hmph.models import Message, Turn
from hmph.source_context import budget_window
from hmph.source_policy import retainable

SLOT_HINTS = {**HINTS, "birthday": ("生日", "birthday"), "goal": ("安排", "计划", "未完成", "plan")}
RAW_HINTS = {
    "name": ("我叫", "叫我", "名字是", "name is"),
    "employer": ("工作", "公司", "work", "employer"),
    "home_city": ("住在", "搬到", "live"),
    "preference": ("喜欢", "爱喝", "prefer"),
}


class RecallPlan(BaseModel):
    intent: Literal["social", "fact", "temporal", "enumeration", "multi_hop"]
    required_slots: tuple[str, ...]
    max_rounds: int = 2
    max_queries: int = 2
    candidate_limit: int = 80
    evidence_limit: int = 12


class FocusSpan(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid", frozen=True)
    source_id: str = Field(min_length=1, max_length=160)
    quote: str = Field(min_length=2, max_length=80)
    slot: Literal["entity", "time", "relation"]


class FocusPlan(BaseModel):
    """One optional literal retrieval bridge, never an answer or inferred fact."""

    model_config = ConfigDict(strict=True, extra="forbid", frozen=True)
    bridge: FocusSpan | None


def focused_query(query, payload, sources):
    """Validate against the supplied first-pass snippets before expanding a query.

    This research primitive is not connected to the online model-call path.
    Product callers must additionally validate source revisions/permission.
    """
    plan = FocusPlan.model_validate(payload)
    if plan.bridge is None:
        return query, None
    bridge = plan.bridge
    original = next((s for s in sources if s.source_id == bridge.source_id), None)
    if original is None or bridge.quote not in original.content:
        raise ValueError("Unverified retrieval bridge")
    lookup = query + " " + bridge.quote
    if len(lookup) > 4000:
        raise ValueError("Retrieval bridge exceeds input limit")
    return lookup, bridge.source_id


class RawEvidence(BaseModel):
    model_config = ConfigDict(frozen=True)
    message_id: UUID
    message_revision: int
    role: str
    recorded_at: datetime
    content: str
    start_offset: int
    end_offset: int
    truncated: bool
    source_timezone: str | None = None
    source_order: int | None = None


class EvidenceSlot(BaseModel):
    name: str
    status: Literal["supported", "conflict", "unknown"]
    memory_refs: tuple[str, ...] = ()
    supported_source_ids: tuple[UUID, ...] = ()
    candidate_source_ids: tuple[UUID, ...] = ()


class EvidenceWorkspace(BaseModel):
    plan: RecallPlan
    slots: tuple[EvidenceSlot, ...]
    rounds: int
    query_count: int
    candidate_count: int
    coverage: Literal["complete", "partial", "unknown"]
    coverage_semantics: str = "retrieved_support_only; raw_candidates_need_reading"
    budget_dropped: int = 0
    backend: str = "lexical"
    retrieval_status: str = "baseline"
    delta_gap: int = 0
    policy_filtered_sources: int = 0
    query_hint_source_ids: tuple[UUID, ...] = ()


def plan_recall(query):
    slots = tuple(p for p, hints in SLOT_HINTS.items() if any(h in query.lower() for h in hints))
    if re.search(r"一共|总共|所有|几次|多少次|how many|all of", query, re.I):
        intent = "enumeration"
    elif PAST.search(query) or re.search(r"\d{4}年|\d{4}[-/]\d{1,2}", query):
        intent = "temporal"
    elif re.search(r"那家|那次|后来|之后.*结果", query):
        intent = "multi_hop"
    elif slots or re.search(r"[?？]|什么|哪个|哪款|何时|记得|上次|when|what|which", query, re.I):
        intent = "fact"
    else:
        intent = "social"
    return RecallPlan(
        intent=intent, required_slots=slots or (() if intent == "social" else ("answer",))
    )


def query_terms(query):
    return list(dict.fromkeys(tokenize(query)))[:32]


def build_workspace(
    plan, memories, raw, short_term, *, query_count, candidate_count, budget_dropped=0
):
    slots = []
    for name in plan.required_slots:
        hits = [hit for hit in memories if hit.predicate == name]
        refs = tuple(f"{hit.id}:{hit.revision}" for hit in hits)
        source_ids = tuple(dict.fromkeys(s.message_id for hit in hits for s in hit.sources))
        if name == "goal" and short_term.data.goal:
            source_ids += (short_term.data.goal.message_id,)
        status = "supported" if refs or source_ids else "unknown"
        if (
            name in FUNCTIONAL
            and plan.intent != "temporal"
            and len({hit.value for hit in hits}) > 1
        ):
            status = "conflict"
        slots.append(
            EvidenceSlot(
                name=name,
                status=status,
                memory_refs=refs,
                supported_source_ids=source_ids,
                candidate_source_ids=tuple(s.message_id for s in raw),
            )
        )
    has_support = any(s.status != "unknown" for s in slots) or bool(raw)
    complete = bool(slots) and all(s.status == "supported" for s in slots)
    if (
        plan.intent in ("enumeration", "temporal", "multi_hop")
        or short_term.pending_gap
        or short_term.state_truncated
        or short_term.coverage_from_seq > 1
    ):
        complete = False
    return EvidenceWorkspace(
        plan=plan,
        slots=tuple(slots),
        rounds=query_count,
        query_count=query_count,
        candidate_count=candidate_count,
        coverage="complete" if complete else "partial" if has_support else "unknown",
        budget_dropped=budget_dropped,
    )


async def search_raw(session, turn, query, slots, *, limit=80):
    terms = list(
        dict.fromkeys(query_terms(query) + [v for p in slots for v in RAW_HINTS.get(p, ())])
    )
    if not terms or limit <= 0:
        return [], 0
    matches = [Message.content.icontains(term, autoescape=True) for term in terms]
    score = sum(case((match, 1), else_=0) for match in matches)
    rows = list(
        await session.execute(
            select(Message, Turn.created_at, Turn.source_timezone)
            .join(Turn)
            .where(
                Turn.pair_id == turn.pair_id,
                Turn.id != turn.id,
                Message.deleted.is_(False),
                or_(Message.role == "user", Turn.status == "completed"),
                or_(*matches),
            )
            .order_by(score.desc(), Turn.created_at.desc(), Message.source_seq.desc(), Message.id)
            .limit(limit)
        )
    )
    ranked = bm25_order([row[0] for row in rows], query)
    ranked_set = set(ranked)
    order = ranked + [i for i in range(len(rows)) if i not in ranked_set]
    evidence = []
    for index in order:
        message, recorded, zone = rows[index]
        if not retainable(message.content):
            continue
        # Excerpts stay contiguous and disclose truncation; they are never new assertions.
        start, end = budget_window(message.content, terms)
        evidence.append(
            RawEvidence(
                message_id=message.id,
                message_revision=message.revision,
                role=message.role,
                recorded_at=recorded.astimezone(ZoneInfo(zone)) if zone else recorded,
                source_timezone=zone,
                content=message.content[start:end],
                start_offset=start,
                end_offset=end,
                truncated=start > 0 or end < len(message.content),
            )
        )
        if len(evidence) == 12:
            break
    return evidence, len(rows)


@dataclass
class WorkingCandidates:
    plan: RecallPlan
    memories: list
    raw: list[RawEvidence]
    query_count: int
    candidate_count: int
    query_hint_source_ids: tuple[UUID, ...] = ()


async def contextual_query(session, turn, query, short_term, preferences):
    """Resolve one deictic lookup from current, grounded state. No new inference.

    A lagging state, explicit correction, multiple possible antecedents or a
    filtered source keeps the original query. Hints locate evidence, not facts.
    """
    if (
        not re.search(r"那家|那次|这件|那个|这事|that (?:company|interview|event)", query, re.I)
        or re.search(r"改成|不是|取消|换到|instead|correction", query, re.I)
        or short_term.pending_gap
        or short_term.processed_through_seq != turn.source_seq - 1
    ):
        return query, ()
    from hmph.companion_policy import SourceCandidate, decide
    from hmph.models import Message, Turn

    closed = {
        item.content.message_id for item in short_term.data.open_items if item.status != "open"
    }
    anchors = [
        a for a in (short_term.data.goal, short_term.data.topic) if a and a.message_id not in closed
    ]
    # A goal takes precedence only when topic and goal name the same source;
    # otherwise ambiguity is visible instead of guessed away.
    unique = {(a.message_id, a.quote): a for a in anchors}
    if len(unique) != 1:
        return query, ()
    anchor = next(iter(unique.values()))
    original = await session.scalar(
        select(Message)
        .join(Turn)
        .where(
            Message.id == anchor.message_id,
            Message.revision == anchor.message_revision,
            Message.deleted.is_(False),
            Message.role == "user",
            Turn.pair_id == turn.pair_id,
        )
    )
    if original is None or anchor.quote not in original.content:
        return query, ()
    policy = decide(
        query,
        preferences,
        [SourceCandidate(original.id, original.content, True)],
        policy_revision=turn.policy_revision,
    )
    if not policy.memory_use or policy.memory_use[0].permission == "omit":
        return query, ()
    if len(query) + len(anchor.quote) + 1 > 4000:
        return query, ()
    return query + " " + anchor.quote, (anchor.message_id,)


async def retrieve(session, turn, query, short_term, *, preferences=None, candidate_limit=80):
    if not 8 <= candidate_limit <= 80:
        raise ValueError("Invalid retrieval candidate limit")
    plan = plan_recall(query)
    lookup_query, hint_sources = await contextual_query(
        session, turn, query, short_term, preferences or {}
    )
    memories = await Memory.search(
        session,
        turn.pair_id,
        lookup_query,
        reference=turn.created_at,
        source_timezone=turn.source_timezone,
    )
    initial = build_workspace(
        plan, memories, [], short_term, query_count=1, candidate_count=len(memories)
    )
    missing = [slot.name for slot in initial.slots if slot.status != "supported"]
    raw, raw_count = [], 0
    rounds = 1
    if missing or plan.intent in ("multi_hop", "enumeration", "temporal"):
        raw, raw_count = await search_raw(
            session, turn, lookup_query, missing, limit=candidate_limit - len(memories)
        )
        rounds = 2
    return WorkingCandidates(plan, memories, raw, rounds, len(memories) + raw_count, hint_sources)
