"""Versioned product memory selection and authoritative evidence preparation.

Native retrieval runs outside PostgreSQL transactions. All returned IDs remain
untrusted until the pair, build, revision, permission and source order agree.
"""

import asyncio
import time
from datetime import datetime
from uuid import UUID
from zoneinfo import ZoneInfo

from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import select, update

from hmph.contracts import ACTIVE, DomainError
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
from hmph.source_policy import retainable
from hmph.working_memory import RawEvidence, WorkingCandidates, plan_recall, query_terms, search_raw

READ_SECONDS = 3
CANDIDATES = 80
SOURCES = 12


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


class ReadTicket(BaseModel):
    model_config = ConfigDict(frozen=True)
    turn_id: UUID
    pair_id: UUID
    build_id: UUID | None
    space_id: UUID | None
    config_revision: int
    build_watermark: int
    cutoff: int
    job_fence: int
    valid_until: datetime


async def pin_read(session, pair, turn, lease):
    if pair.selected_memory_build is None:
        return None
    turn.memory_context = {
        "backend": "lexical",
        "retrieval_status": "build_unavailable",
        "config_revision": pair.memory_config_revision,
    }
    ready = await ready_space(session, pair, pair.selected_memory_build)
    build, space = ready if ready else (None, None)
    ticket = ReadTicket(
        turn_id=turn.id,
        pair_id=pair.id,
        build_id=build.id if build else None,
        space_id=space.id if space else None,
        config_revision=pair.memory_config_revision,
        build_watermark=build.watermark if build else 0,
        cutoff=pair.next_memory_seq - 1,
        job_fence=lease.fence,
        valid_until=turn.deadline,
    )
    if ready:
        session.add(MemoryRead(**ticket.model_dump(exclude={"pair_id", "build_watermark"})))
    turn.memory_context = {
        "backend": "graphiti",
        "retrieval_status": "preparing",
        **ticket.model_dump(mode="json", exclude={"pair_id", "turn_id"}),
    }
    return ticket


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
            return result, "ok" if result.source_ids else "empty"
    except TimeoutError:
        return NativeSources(requests=None), "timeout"
    except SourceRetrievalError as error:
        return NativeSources(requests=0 if error.code == "reader_recovering" else None), error.code
    except Exception:
        # Native exceptions may contain query text, database passwords or data.
        return NativeSources(requests=None), "reader_failed"


async def expand_sources(session, turn, ticket, native, status, query, *, budget):
    """Prioritize complete post-build input before older retrieved evidence.

    An incomplete delta suppresses all pre-build evidence. This is deliberately
    conservative until structured correction/subject resolution is validated.
    """
    current = await session.scalar(
        select(Message.id).where(Message.turn_id == turn.id, Message.role == "user")
    )
    delta = list(
        (
            await session.execute(
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
                .limit(CANDIDATES + 1)
            )
        ).all()
    )
    scan_overflow = len(delta) > CANDIDATES
    delta_gap = int(scan_overflow)
    delta = delta[:CANDIDATES]
    # Reserve half of the shared candidate capacity for exact original-text
    # matches. A successful semantic search can still omit the decisive source.
    native_capacity = max(0, (CANDIDATES - len(delta)) // 2)
    source_ids = tuple(dict.fromkeys(native.source_ids))[:native_capacity]
    rows = list(
        (
            await session.execute(
                select(MemoryNativeSource, Message, Turn, MemoryBuildSource)
                .join(Message, MemoryNativeSource.message_id == Message.id)
                .join(Turn, Message.turn_id == Turn.id)
                .join(
                    MemoryBuildSource,
                    (MemoryBuildSource.build_id == ticket.build_id)
                    & (MemoryBuildSource.message_id == Message.id),
                )
                .where(
                    MemoryNativeSource.space_id == ticket.space_id,
                    MemoryNativeSource.native_id.in_([str(i) for i in source_ids]),
                )
            )
        ).all()
    )
    valid = len(rows) == len(source_ids)
    for mapping, message, source_turn, frozen in rows:
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
    selected, used, filtered, truncated = [], 0, 0, 0
    terms = query_terms(query)
    for message, source_turn, task in delta:
        if not retainable(message.content):
            filtered += 1
            # Omitted boundaries must not leave pre-build values in context.
            continue
        source = _raw(message, source_turn, task.source_order)
        size = len(source.model_dump_json().encode()) + 64
        if used + size > budget or len(selected) >= SOURCES:
            delta_gap += 1
            continue
        selected.append(source)
        used += size
    # Reuse the existing PostgreSQL source lookup as a complement, within the
    # same candidate/deadline/context budgets. This adds no framework writer.
    fallback = []
    fallback_scanned = 0
    if not delta_gap and not filtered:
        fallback, fallback_scanned = await search_raw(
            session, turn, query, (), limit=CANDIDATES - len(delta) - len(rows)
        )
    candidates = [_raw(m, t, mapping.source_order) for mapping, m, t, _ in rows]
    native_messages = {s.message_id for s in candidates}
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
        # A fallback match still needs pair-global source order and permission.
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
        start, end = budget_window(message.content, terms)
        source = source.model_copy(
            update={
                "content": message.content[start:end],
                "start_offset": start,
                "end_offset": end,
                "truncated": start > 0 or end < len(message.content),
            }
        )
        size = len(source.model_dump_json().encode()) + 64
        if used + size <= budget:
            selected.append(source)
            seen.add(source.message_id)
            used += size
            truncated += int(source.truncated)
    # Presentation order reflects observation order, never event-time sorting.
    selected.sort(key=lambda s: s.source_order)
    audit = {
        "backend": "graphiti"
        if status == "ok" and any(m.id in {s.message_id for s in selected} for _, m, _, _ in rows)
        else "lexical",
        "retrieval_status": status,
        "delta_gap": delta_gap,
        "delta_scanned": len(delta),
        "delta_scan_truncated": scan_overflow,
        "delta_scanned_from_order": min((task.source_order for _, _, task in delta), default=None),
        "native_budget_dropped": len(set(native.source_ids)) - len(source_ids),
        "lexical_complement_sources": sum(
            s.message_id not in native_messages and s.source_order <= ticket.build_watermark
            for s in selected
        )
        if status == "ok"
        else 0,
        "lexical_candidates_scanned": fallback_scanned,
        "filtered_sources": filtered,
        "truncated_sources": truncated,
        "requests": native.requests,
        "database_queries": native.database_queries,
        "build_watermark": ticket.build_watermark,
        "cutoff": ticket.cutoff,
        "config_revision": ticket.config_revision,
        "build_id": str(ticket.build_id) if ticket.build_id else None,
        "space_id": str(ticket.space_id) if ticket.space_id else None,
    }
    return WorkingCandidates(
        plan_recall(query),
        [],
        selected,
        native.requests if native.requests is not None else 1,
        len(rows) + len(delta) + fallback_scanned,
    ), audit


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
                MemoryBuild.backend == "graphiti",
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
    def __init__(self, db, *, digest_key):
        self.db, self.ledger = db, RequestLedger(digest_key)

    async def _pair(self, session, owner_id, character_id, *, lock=False):
        query = select(Pair).where(Pair.owner_id == owner_id, Pair.character_id == character_id)
        pair = await session.scalar(query.with_for_update() if lock else query)
        if pair is None:
            raise DomainError("not_found", 404)
        return pair

    async def _view(self, session, pair):
        build = pair.selected_memory_build
        return SelectionView(
            mode="graphiti_experimental" if build else "lexical",
            revision=pair.memory_config_revision,
            build_id=build,
            available=build is None or await ready_space(session, pair, build) is not None,
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
            if build_id:
                from sqlalchemy import func

                from hmph.memory_builds import MemoryBuilds

                build, space = await ready_space(session, pair, build_id)
                await MemoryBuilds._sources(session, build)
                mapped = await session.scalar(
                    select(func.count())
                    .select_from(MemoryNativeSource)
                    .where(MemoryNativeSource.space_id == space.id)
                )
                if mapped != build.source_count:
                    raise DomainError("memory_build_unavailable")
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
            self.ledger.save(session, owner_id, "memory.select", request_id, digest, pair.id)
            await session.flush()
            return await self._view(session, pair)
