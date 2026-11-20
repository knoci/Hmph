"""Versioned product memory selection and authoritative evidence preparation.

Native retrieval runs outside PostgreSQL transactions. All returned IDs remain
untrusted until the pair, build, revision, permission and source order agree.
"""

import asyncio
import re
import time
from datetime import datetime
from typing import Literal
from uuid import UUID
from zoneinfo import ZoneInfo

from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import case, or_, select, update

from hmph.contracts import ACTIVE, DomainError
from hmph.event_changes import WITHDRAWAL, asks_current_state, withdrawal_focus, withdrawal_window
from hmph.lexical import tokenize
from hmph.memory import Memory
from hmph.models import (
    MemoryBuild,
    MemoryBuildSource,
    MemoryNativeSource,
    MemoryRead,
    MemorySpace,
    MemoryTask,
    Message,
    Pair,
    Turn,
)
from hmph.persistence import RequestLedger, terminate_turn
from hmph.source_context import budget_window
from hmph.source_policy import NO_RETAIN, SECRET, retainable
from hmph.working_memory import RawEvidence, WorkingCandidates, plan_recall, query_terms, search_raw

READ_SECONDS = 3
CANDIDATES = 80
SOURCES = 12
UPDATE_CUE = re.compile(
    r"更正|纠正|改[成为到期了]|换[成到了]|更新|延期|推迟|提前|取消|不再|不去了|"
    r"不是[^。！？!?\n]{0,60}是|现在是|定在|"
    r"\b(?:correction|actually|instead|no longer|"
    r"updat(?:e|ed)|chang(?:e|ed)|mov(?:e|ed)|"
    r"reschedul(?:e|ed)|cancel(?:led|ed))\b|" + WITHDRAWAL,
    re.I,
)
SENTENCES = re.compile(r"[^。！？.!?\n]+[。！？.!?]?")


class SourceRetrievalError(Exception):
    def __init__(self, code):
        if code not in {"reader_failed", "reader_recovering"}:
            raise ValueError("Invalid retrieval error code")
        self.code = code
        super().__init__(code)


class NativeSources(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    source_ids: tuple[UUID, ...] = Field(default=(), max_length=CANDIDATES)
    requests: int | None = Field(default=1, ge=0, le=4)
    database_queries: int | None = Field(default=None, ge=0, le=32)
    branch_status: dict[Literal["mem0", "timem"], str] = Field(default_factory=dict)
    timem_sources: tuple[UUID, ...] = Field(default=(), max_length=CANDIDATES)
    timem_strategy: (
        Literal[
            "hierarchical", "linked_exchanges", "originals_and_hierarchy", "episodes_and_hierarchy"
        ]
        | None
    ) = None


class ReadTicket(BaseModel):
    model_config = ConfigDict(frozen=True)
    turn_id: UUID
    pair_id: UUID
    build_id: UUID | None
    space_id: UUID | None
    config_revision: int
    build_watermark: int
    backend: Literal["graphiti", "mem0", "mem0_timem"] = "graphiti"
    cutoff: int
    job_fence: int
    valid_until: datetime
    reference_time: datetime | None = Field(default=None, exclude=True)
    source_timezone: str | None = Field(default=None, exclude=True)


async def pin_read(session, pair, turn, lease, *, advance_timem=False):
    if pair.selected_memory_build is None:
        return None
    if advance_timem:
        await _adopt_timem_maintenance(session, pair)
    turn.memory_context = {
        "backend": "lexical",
        "retrieval_status": "build_unavailable",
        "config_revision": pair.memory_config_revision,
    }
    ready = await ready_space(session, pair, pair.selected_memory_build)
    build, space = ready if ready else (None, None)
    selected = build or await session.get(MemoryBuild, pair.selected_memory_build)
    ticket = ReadTicket(
        turn_id=turn.id,
        pair_id=pair.id,
        build_id=build.id if build else None,
        space_id=space.id if space else None,
        config_revision=pair.memory_config_revision,
        build_watermark=build.watermark if build else 0,
        backend=selected.backend if selected else "graphiti",
        cutoff=pair.next_memory_seq - 1,
        job_fence=lease.fence,
        valid_until=turn.deadline,
        reference_time=turn.created_at,
        source_timezone=turn.source_timezone,
    )
    if ready:
        session.add(
            MemoryRead(**ticket.model_dump(exclude={"pair_id", "build_watermark", "backend"}))
        )
    turn.memory_context = {
        "retrieval_status": "preparing",
        **ticket.model_dump(mode="json", exclude={"pair_id", "turn_id"}),
    }
    return ticket


async def _adopt_timem_maintenance(session, pair):
    """Select a finished maintenance generation before pinning this pair's next turn."""
    from hmph.memory_builds import MemoryBuilds
    from hmph.memory_maintenance import MaintenancePolicy

    policy = MaintenancePolicy.model_validate(pair.memory_maintenance)
    if not policy.enabled or policy.backend != "mem0_timem":
        return
    previous = await session.get(MemoryBuild, pair.selected_memory_build)
    if previous is None or previous.backend != "mem0_timem":
        return
    minimum_watermark = previous.watermark if previous.memory_epoch == pair.memory_epoch else 0
    candidate = await session.scalar(
        select(MemoryBuild)
        .join(MemorySpace)
        .where(
            MemoryBuild.pair_id == pair.id,
            MemoryBuild.backend == previous.backend,
            MemoryBuild.generation > previous.generation,
            MemoryBuild.watermark >= minimum_watermark,
            MemoryBuild.memory_epoch == pair.memory_epoch,
            MemoryBuild.maintenance_revision == policy.revision,
            MemoryBuild.status == "ready",
            MemorySpace.status == "ready",
            MemorySpace.quiesced.is_(True),
        )
        .order_by(MemoryBuild.generation.desc())
        .limit(1)
    )
    if candidate is None:
        return
    pair.selected_memory_build = candidate.id
    pair.memory_config_revision += 1
    if previous.status == "ready":
        await MemoryBuilds._retire(session, [previous.id])


async def validate_read(session, pair, turn, job, now):
    context = turn.memory_context
    if context is None:
        return
    if context.get("config_revision") != pair.memory_config_revision:
        raise DomainError("stale_lease")
    if not context.get("build_id"):
        return
    read = await session.get(MemoryRead, turn.id)
    if read is None:
        raise DomainError("stale_lease")
    ready = await ready_space(session, pair, read.build_id)
    if (
        read.invalidated
        or read.valid_until <= now
        or read.job_fence != job.fence
        or read.config_revision != pair.memory_config_revision
        or pair.selected_memory_build != read.build_id
        or ready is None
        or ready[1].id != read.space_id
    ):
        raise DomainError("stale_lease")


async def retrieve_native(reader, ticket, query, *, deadline):
    if ticket.space_id is None:
        return NativeSources(requests=0), "build_unavailable"
    if reader is None:
        return NativeSources(requests=0), "reader_unavailable"
    try:
        async with asyncio.timeout(max(0, deadline - time.monotonic())):
            result = NativeSources.model_validate(await reader.retrieve_sources(ticket, query))
            if result.source_ids:
                return result, "ok"
            branches = set(result.branch_status.values())
            if branches <= {"empty"}:
                return result, "empty"
            if "reader_recovering" in branches:
                return result, "reader_recovering"
            if branches <= {"empty", "timeout"}:
                return result, "timeout"
            return result, "reader_failed"
    except TimeoutError:
        return NativeSources(requests=None), "timeout"
    except SourceRetrievalError as error:
        return NativeSources(requests=0 if error.code == "reader_recovering" else None), error.code
    except Exception:
        # Native exceptions may contain query text, database passwords or data.
        return NativeSources(requests=None), "reader_failed"


async def expand_sources(
    session, turn, ticket, native, status, query, *, budget, lookup_query=None
):
    """Scan post-build input before admitting older retrieved evidence.

    TiMem + mem0 can narrow a fully scanned delta under context pressure. Keep
    matching originals and possible updates. A scan gap or dropped candidate
    still suppresses pre-build evidence.
    """
    current = await session.scalar(
        select(Message.id).where(Message.turn_id == turn.id, Message.role == "user")
    )
    focus_query = lookup_query or query
    terms = query_terms(focus_query)
    plan = plan_recall(query)
    # Native hits locate original messages. Keep independently committed facts
    # available both alongside those hits and after a native snapshot retires.
    memories = []
    if ticket.backend in {"mem0", "mem0_timem"}:
        memories = await Memory.search(
            session,
            turn.pair_id,
            lookup_query or query,
            reference=turn.created_at,
            source_timezone=turn.source_timezone,
        )
    native_ids = tuple(dict.fromkeys(native.source_ids))
    delta_limit = CANDIDATES - len(memories) - min(SOURCES, len(native_ids))
    focused_delta = (
        ticket.backend == "mem0_timem"
        and (status != "timeout" or bool(native.branch_status))
        and bool(terms)
        and ticket.cutoff - ticket.build_watermark - 1 > delta_limit
    )
    delta_query = (
        select(Message, Turn, MemoryTask)
        .join(Turn, Message.turn_id == Turn.id)
        .join(MemoryTask, MemoryTask.source_message_id == Message.id)
        .where(
            MemoryTask.pair_id == ticket.pair_id,
            Turn.pair_id == ticket.pair_id,
            MemoryTask.source_order > ticket.build_watermark,
            MemoryTask.source_order <= ticket.cutoff,
            Message.id != current,
            Message.role == "user",
            Message.deleted.is_(False),
            MemoryTask.source_revision == Message.revision,
        )
        .order_by(MemoryTask.source_order.desc())
    )
    if focused_delta:
        # Match throughout the interval before applying the candidate limit.
        # PostgreSQL spells a word boundary \y. Include exclusions so a
        # source we cannot use still signals a gap rather than disappearing.
        patterns = (UPDATE_CUE, SECRET, NO_RETAIN)
        topical = or_(*(Message.content.icontains(term, autoescape=True) for term in terms))
        delta_query = (
            delta_query.where(
                or_(
                    topical,
                    *(Message.content.op("~*")(p.pattern.replace(r"\b", r"\y")) for p in patterns),
                )
            )
            .order_by(None)
            .order_by(case((topical, 1), else_=0).desc(), MemoryTask.source_order.desc())
        )
    else:
        delta_limit = CANDIDATES
    delta = (await session.execute(delta_query.limit(delta_limit + 1))).all()
    scan_overflow = len(delta) > delta_limit
    delta_gap = int(scan_overflow)
    delta = delta[:delta_limit]
    native_capacity = max(0, CANDIDATES - len(delta) - len(memories))
    if ticket.backend == "graphiti":
        native_capacity //= 2  # Reserve space for Graphiti's existing lexical complement.
    source_ids = native_ids[:native_capacity]
    rows = list(
        (
            await session.execute(
                select(
                    MemoryNativeSource, Message, Turn, MemoryBuildSource, MemoryTask.source_order
                )
                .join(Message, MemoryNativeSource.message_id == Message.id)
                .join(Turn, Message.turn_id == Turn.id)
                .join(
                    MemoryBuildSource,
                    (MemoryBuildSource.build_id == ticket.build_id)
                    & (MemoryBuildSource.message_id == Message.id),
                )
                .outerjoin(
                    MemoryTask,
                    (MemoryTask.source_message_id == Message.id)
                    & (MemoryTask.source_revision == Message.revision)
                    & (MemoryTask.pair_id == ticket.pair_id)
                    & (MemoryTask.source_order == MemoryNativeSource.source_order),
                )
                .where(
                    MemoryNativeSource.space_id == ticket.space_id,
                    MemoryNativeSource.native_id.in_([str(i) for i in source_ids]),
                )
            )
        ).all()
    )
    valid = len(rows) == len(source_ids)
    for mapping, message, source_turn, frozen, _ in rows:
        if (
            source_turn.pair_id != ticket.pair_id
            or message.deleted
            or message.role != "user"
            or message.revision != mapping.message_revision
            or frozen.message_revision != mapping.message_revision
            or frozen.source_order != mapping.source_order
            or mapping.source_order > min(ticket.build_watermark, ticket.cutoff)
            or source_turn.created_at > turn.created_at
            or not retainable(message.content)
        ):
            valid = False
    if not valid:
        status, rows = "invalid_sources", []
    else:
        native_rank = {str(source_id): index for index, source_id in enumerate(source_ids)}
        rows.sort(key=lambda row: native_rank[row[0].native_id])
    candidates = [_raw(m, t, order) for _, m, t, _, order in rows if order is not None]
    scope, conditions, changes = set(), {}, ()
    if (
        ticket.backend == "mem0_timem"
        and plan.intent in {"fact", "multi_hop"}
        and asks_current_state(query)
    ):
        focus = withdrawal_focus(
            candidates
            + [_raw(m, t, task.source_order) for m, t, task in delta if retainable(m.content)],
            focus_query,
        )
        scope, conditions, changes = focus.terms, focus.conditions, focus.changes
    out_of_scope = 0
    selected, used, filtered, truncated = [], 0, 0, 0
    delta_sources = []
    delta_unselected = None if focused_delta else 0
    recovery_social = (
        ticket.backend == "mem0_timem"
        and status == "build_unavailable"
        and plan.intent == "social"
    )
    for message, source_turn, task in delta:
        if not retainable(message.content):
            filtered += 1
            # Omitted boundaries must not leave pre-build values in context.
            continue
        if recovery_social and source_turn.conversation_id != turn.conversation_id:
            content = message.content.lower()
            if not terms or sum(term in content for term in terms) < min(2, len(terms)):
                # An unfiltered recent window can replay an unrelated chat as
                # evidence while the replacement snapshot is being built.
                if delta_unselected is not None:
                    delta_unselected += 1
                continue
        source = _raw(message, source_turn, task.source_order)
        if scope:
            source = _scope_withdrawal(source, scope, focus_query, conditions)
            if source is None:
                out_of_scope += 1
                continue
        size = len(source.prompt_json().encode()) + 64
        delta_sources.append((source, size))
    recovery_scope = status == "build_unavailable" and plan.intent in {"fact", "multi_hop"}
    if (
        ticket.backend == "mem0_timem"
        and terms
        and not scan_overflow
        and not focused_delta
        and (
            recovery_scope
            or len(delta_sources) >= SOURCES
            or sum(size for _, size in delta_sources) > budget // 2
        )
    ):
        scoped = []
        required_matches = min(2, len(terms)) if recovery_scope else 1
        for source, size in delta_sources:
            text = source.content.lower()
            update = UPDATE_CUE.search(text)
            # A prior question about another subject isn't a correction merely
            # because it contains a state cue such as "现在是".
            if recovery_scope and text.rstrip().endswith(("？", "?")):
                update = None
            if update or sum(term in text for term in terms) >= required_matches:
                scoped.append((source, size))
        delta_unselected += len(delta_sources) - len(scoped)
        delta_sources = scoped
    delta_truncated = 0
    for source, size in delta_sources:
        if ticket.backend == "mem0_timem" and used + size > budget and len(selected) < SOURCES:
            excerpt = _correction_excerpt(source, budget - used)
            if excerpt is not None:
                source, size = excerpt
                delta_truncated += 1
        if used + size > budget or len(selected) >= SOURCES:
            delta_gap += 1
            continue
        selected.append(source)
        used += size
    # Reuse the existing PostgreSQL source lookup as a complement, within the
    # same candidate/deadline/context budgets. This adds no framework writer.
    fallback = []
    fallback_scanned = 0
    if not delta_gap and not filtered and ticket.backend == "graphiti":
        fallback, fallback_scanned = await search_raw(
            session, turn, query, (), limit=CANDIDATES - len(delta) - len(rows)
        )
    if changes:
        ordered = dict.fromkeys((*changes[:2], *tuple(conditions)[:2]))
        priority = {source_id: rank for rank, source_id in enumerate(ordered)}
        candidates.sort(key=lambda source: priority.get(source.message_id, len(priority)))
    elif (
        ticket.backend == "mem0_timem"
        and native.branch_status.get("timem") == "ok"
        and not scope
        and plan.intent in {"fact", "multi_hop"}
        and asks_current_state(query)
    ):
        candidates = _prefer_bridged_update(candidates, focus_query)
    native_messages = {s.message_id: s for s in candidates}
    if delta_gap or filtered:
        candidates = []
    else:
        candidates = [
            items[i]
            for i in range(max(len(candidates), len(fallback)))
            for items in (fallback, candidates)
            if i < len(items)
        ]
    seen = {s.message_id for s in selected}
    for source in candidates:
        if source.message_id in seen or len(selected) >= SOURCES:
            continue
        if source.message_id in native_messages:
            source = native_messages[source.message_id]
        else:
            # Graphiti's complementary raw lookup still needs the pair-global order.
            task = await session.scalar(
                select(MemoryTask).where(
                    MemoryTask.source_message_id == source.message_id,
                    MemoryTask.source_revision == source.message_revision,
                    MemoryTask.pair_id == ticket.pair_id,
                    MemoryTask.source_order <= ticket.cutoff,
                )
            )
            if task is None or source.role != "user":
                continue
            message = await session.get(Message, source.message_id)
            if message.deleted or not retainable(message.content):
                continue
            source = _raw(message, await session.get(Turn, message.turn_id), task.source_order)
        if scope:
            source = _scope_withdrawal(source, scope, focus_query, conditions)
            if source is None:
                out_of_scope += 1
                continue
        start, end = budget_window(source.content, terms)
        source = source.model_copy(
            update={
                "content": source.content[start:end],
                "start_offset": source.start_offset + start,
                "end_offset": source.start_offset + end,
                "truncated": source.truncated or start > 0 or end < len(source.content),
            }
        )
        size = len(source.prompt_json().encode()) + 64
        if used + size <= budget:
            selected.append(source)
            seen.add(source.message_id)
            used += size
            truncated += int(source.truncated)
    # Presentation order reflects observation order, never event-time sorting.
    selected.sort(key=lambda s: s.source_order)
    selected_native = native_messages.keys() & {s.message_id for s in selected}
    audit = {
        "backend": ("mem0_timem" if native.branch_status else ticket.backend)
        if status == "ok" and selected_native
        else "lexical"
        if ticket.backend == "graphiti"
        else "canonical"
        if memories or (focused_delta and selected)
        else "recent_only",
        "retrieval_status": status,
        "delta_gap": delta_gap,
        "delta_scanned": len(delta),
        "delta_scope": "matched" if focused_delta else "recent",
        "delta_unselected": delta_unselected,
        "delta_truncated": delta_truncated,
        "delta_scan_truncated": scan_overflow,
        "delta_scanned_from_order": None
        if focused_delta
        else min((task.source_order for _, _, task in delta), default=None),
        "native_budget_dropped": len(set(native.source_ids)) - len(source_ids),
        "lexical_complement_sources": sum(
            s.message_id not in native_messages and s.source_order <= ticket.build_watermark
            for s in selected
        )
        if status == "ok"
        else 0,
        "lexical_candidates_scanned": fallback_scanned,
        "filtered_sources": filtered,
        "withdrawal_scope_skipped": out_of_scope,
        "truncated_sources": truncated + delta_truncated,
        "requests": native.requests,
        "database_queries": native.database_queries,
        "branch_status": native.branch_status,
        "timem_source_ids": [str(s) for s in native.timem_sources if s in selected_native],
        "timem_strategy": native.timem_strategy,
        "build_watermark": ticket.build_watermark,
        "cutoff": ticket.cutoff,
        "config_revision": ticket.config_revision,
        "build_id": str(ticket.build_id) if ticket.build_id else None,
        "space_id": str(ticket.space_id) if ticket.space_id else None,
    }
    return WorkingCandidates(
        plan,
        memories,
        selected,
        native.requests if native.requests is not None else 1,
        min(CANDIDATES, len(rows) + len(delta) + fallback_scanned + len(memories)),
    ), audit


def _prefer_bridged_update(candidates, query):
    """Keep the answer between two on-topic turns in one conversation."""
    terms = set(tokenize(query))
    matching = {
        (source.conversation_id, source.source_order): source
        for source in candidates
        if source.conversation_id and len(terms.intersection(tokenize(source.content))) >= 2
    }
    for index, source in enumerate(candidates):
        previous = matching.get((source.conversation_id, source.source_order - 1))
        following = matching.get((source.conversation_id, source.source_order + 1))
        if previous and following and UPDATE_CUE.search(following.content):
            return [source, *candidates[:index], *candidates[index + 1:]]
    return candidates


def _scope_withdrawal(source, terms, query, conditions):
    span = conditions.get(source.message_id) or withdrawal_window(source.content, terms, query)
    if span is None:
        return None
    start, end = span
    return source.model_copy(
        update={
            "content": source.content[start:end],
            "start_offset": source.start_offset + start,
            "end_offset": source.start_offset + end,
            "truncated": source.truncated or start > 0 or end < len(source.content),
        }
    )


def _correction_excerpt(source, available_bytes):
    """Keep all update-bearing sentences and their immediate context as one exact span."""
    sentences = list(SENTENCES.finditer(source.content))
    updates = [i for i, sentence in enumerate(sentences) if UPDATE_CUE.search(sentence.group())]
    if not updates:
        return None
    start = sentences[max(0, updates[0] - 1)].start()
    end = sentences[min(len(sentences) - 1, updates[-1] + 1)].end()
    if end - start == len(source.content):
        return None
    excerpt = source.model_copy(
        update={
            "content": source.content[start:end],
            "start_offset": source.start_offset + start,
            "end_offset": source.start_offset + end,
            "truncated": True,
        }
    )
    size = len(excerpt.prompt_json().encode()) + 64
    return (excerpt, size) if size <= available_bytes else None


def _raw(message, turn, source_order):
    return RawEvidence(
        message_id=message.id,
        message_revision=message.revision,
        role=message.role,
        recorded_at=(
            turn.created_at.astimezone(ZoneInfo(turn.source_timezone))
            if turn.source_timezone
            else turn.created_at
        ),
        content=message.content,
        start_offset=0,
        end_offset=len(message.content),
        truncated=False,
        source_timezone=turn.source_timezone,
        source_order=source_order,
        conversation_id=turn.conversation_id,
    )


class SelectionView(BaseModel):
    mode: str
    revision: int
    build_id: UUID | None
    available: bool


async def ready_space(session, pair, build_id):
    return (
        await session.execute(
            select(MemoryBuild, MemorySpace)
            .join(MemorySpace)
            .where(
                MemoryBuild.id == build_id,
                MemoryBuild.pair_id == pair.id,
                MemoryBuild.backend.in_(("graphiti", "mem0", "mem0_timem")),
                MemoryBuild.status == "ready",
                MemoryBuild.memory_epoch == pair.memory_epoch,
                MemorySpace.status == "ready",
                MemorySpace.quiesced.is_(True),
            )
        )
    ).first()


async def invalidate_reads(session, build_ids):
    await session.execute(
        update(MemoryRead).where(MemoryRead.build_id.in_(build_ids)).values(invalidated=True)
    )
    turns = list(
        await session.scalars(
            select(Turn)
            .join(MemoryRead)
            .where(MemoryRead.build_id.in_(build_ids), Turn.status.in_(ACTIVE))
            .with_for_update(of=Turn)
        )
    )
    for turn in turns:
        await terminate_turn(session, turn, "invalidated", "memory_build_retired")


class MemorySelection:
    def __init__(self, db, *, digest_key, allow_mem0=False, allow_timem=False):
        self.db, self.ledger = db, RequestLedger(digest_key)
        self.allow_mem0 = allow_mem0
        self.allow_timem = allow_timem

    async def _pair(self, session, owner_id, character_id, *, lock=False):
        query = select(Pair).where(Pair.owner_id == owner_id, Pair.character_id == character_id)
        pair = await session.scalar(query.with_for_update() if lock else query)
        if pair is None:
            raise DomainError("not_found", 404)
        return pair

    async def _view(self, session, pair):
        build_id = pair.selected_memory_build
        build = await session.get(MemoryBuild, build_id) if build_id else None
        return SelectionView(
            mode=f"{build.backend}_experimental" if build else "lexical",
            revision=pair.memory_config_revision,
            build_id=build_id,
            available=build_id is None
            or (
                (build is None or build.backend != "mem0_timem" or self.allow_timem)
                and await ready_space(session, pair, build_id) is not None
            ),
        )

    async def get(self, owner_id, character_id):
        async with self.db.sessions() as session:
            return await self._view(session, await self._pair(session, owner_id, character_id))

    async def update(self, owner_id, character_id, request_id, *, expected_revision, build_id):
        digest = self.ledger.digest(
            str(character_id),
            {
                "expected_revision": expected_revision,
                "build_id": str(build_id) if build_id else None,
            },
        )
        async with self.db.sessions.begin() as session:
            await self.ledger.owner_lock(session, owner_id)
            replay = await self.ledger.request(
                session, owner_id, "memory.select", request_id, digest
            )
            pair = await self._pair(session, owner_id, character_id, lock=True)
            if replay:
                return await self._view(session, pair)
            if pair.memory_config_revision != expected_revision:
                raise DomainError("memory_config_conflict")
            if build_id and await ready_space(session, pair, build_id) is None:
                raise DomainError("memory_build_unavailable")
            superseded = []
            if build_id:
                from sqlalchemy import func

                from hmph.memory_builds import MemoryBuilds

                build, space = await ready_space(session, pair, build_id)
                if (build.backend == "mem0" and not self.allow_mem0) or (
                    build.backend == "mem0_timem" and not self.allow_timem
                ):
                    raise DomainError("memory_build_unavailable")
                await MemoryBuilds._sources(session, build)
                mapped = await session.scalar(
                    select(func.count())
                    .select_from(MemoryNativeSource)
                    .where(MemoryNativeSource.space_id == space.id)
                )
                if mapped != build.source_count:
                    raise DomainError("memory_build_unavailable")
                if build.backend == "mem0_timem":
                    superseded = list(
                        await session.scalars(
                            select(MemoryBuild.id).where(
                                MemoryBuild.pair_id == pair.id,
                                MemoryBuild.backend == build.backend,
                                MemoryBuild.status == "ready",
                                MemoryBuild.generation < build.generation,
                            )
                        )
                    )
            pair.selected_memory_build = build_id
            pair.memory_config_revision += 1
            turns = list(
                await session.scalars(
                    select(Turn)
                    .where(Turn.pair_id == pair.id, Turn.status.in_(ACTIVE))
                    .with_for_update()
                )
            )
            for turn in turns:
                read = await session.get(MemoryRead, turn.id)
                if read:
                    read.invalidated = True
                await terminate_turn(session, turn, "invalidated", "memory_config_changed")
            if superseded:
                await MemoryBuilds._retire(session, superseded)
            self.ledger.save(session, owner_id, "memory.select", request_id, digest, pair.id)
            await session.flush()
            return await self._view(session, pair)
