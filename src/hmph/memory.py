"""Durable extractive memory. All writes are serialized by the pair lock.

Model output is a proposal; the commit checks its source, revision and lease.
The initial reader is intentionally a lexical baseline, not semantic retrieval.
"""

import re
from datetime import timedelta
from uuid import UUID

from sqlalchemy import and_, case, delete, func, or_, select, update

from hmph.contracts import ACTIVE, DomainError
from hmph.database import Database
from hmph.dialogue import invalidate_dialogue
from hmph.lexical import tokenize
from hmph.memory_builds import framework_space_count, retire_for_forget
from hmph.memory_types import (
    DeletionReceipt,
    ForgetPreview,
    LearnResult,
    MemoryLease,
    MemoryProposal,
    MemoryTaskView,
    MemoryView,
    ReferencedTime,
    SourceView,
)
from hmph.models import (
    DialogueSource,
    MemoryDeletion,
    MemoryEntry,
    MemorySource,
    MemoryTask,
    MemoryVersion,
    Message,
    Pair,
    StreamEvent,
    Turn,
    TurnMemoryUse,
    TurnSourceUse,
)
from hmph.persistence import RequestLedger, append_event, terminate_turn
from hmph.source_policy import retainable
from hmph.temporal import normalize_time, query_intervals

FUNCTIONAL = {"name", "employer", "home_city"}
HINTS = {
    "name": ("叫什么", "名字", "称呼", "姓名", "my name"),
    "employer": ("公司", "工作", "雇主", "employer", "work"),
    "home_city": ("住哪", "住在", "城市", "home", "live"),
    "preference": ("喜欢", "偏好", "爱喝", "饮料", "口味", "prefer"),
    "trip": ("旅行", "旅游", "去过", "travel", "trip"),
}
PAST = re.compile(r"以前|之前|过去|曾经|去年|上个月|上周|当时|previous|before|used to", re.I)


class Memory:
    def __init__(self, db: Database, *, digest_key: bytes, source_timezone="UTC"):
        self.db = db
        self.digest_key = digest_key
        self.source_timezone = source_timezone
        self.ledger = RequestLedger(digest_key)

    async def claim(self, worker_id: str, lease_seconds: float = 30, *, model_mode="fake"):
        if not 1 <= len(worker_id) <= 100 or lease_seconds <= 0:
            raise ValueError("Invalid memory lease settings")
        async with self.db.sessions.begin() as session:
            busy = select(MemoryTask.pair_id).where(MemoryTask.status == "leased")
            pair = await session.scalar(
                select(Pair)
                .join(MemoryTask, MemoryTask.pair_id == Pair.id)
                .where(
                    MemoryTask.status == "pending",
                    MemoryTask.model_mode == model_mode,
                    Pair.id.not_in(busy),
                )
                .order_by(MemoryTask.source_order, MemoryTask.id)
                .limit(1)
                .with_for_update(of=Pair, skip_locked=True)
            )
            if pair is None:
                return None
            # Reread the queue after acquiring its serialization lock.
            if await session.scalar(
                select(MemoryTask.id)
                .where(MemoryTask.pair_id == pair.id, MemoryTask.status == "leased")
                .limit(1)
            ):
                return None
            task = await session.scalar(
                select(MemoryTask)
                .where(
                    MemoryTask.pair_id == pair.id,
                    MemoryTask.status == "pending",
                    MemoryTask.model_mode == model_mode,
                )
                .order_by(MemoryTask.source_order)
                .limit(1)
                .with_for_update()
            )
            if task is None:
                return None
            source = await session.get(Message, task.source_message_id)
            if (
                task.memory_epoch != pair.memory_epoch
                or source.deleted
                or source.revision != task.source_revision
                or not retainable(source.content)
            ):
                task.status, task.error_code = "done", "source_ineligible"
                return None
            now = await session.scalar(select(func.clock_timestamp()))
            task.status, task.worker_id = "leased", worker_id
            task.fence += 1
            task.attempts += 1
            task.lease_until = now + timedelta(seconds=lease_seconds)
            return MemoryLease(
                id=task.id,
                pair_id=pair.id,
                fence=task.fence,
                worker_id=worker_id,
                memory_epoch=pair.memory_epoch,
                content=source.content,
            )

    async def _lease(self, session, lease):
        pair = await session.scalar(select(Pair).where(Pair.id == lease.pair_id).with_for_update())
        task = await session.get(MemoryTask, lease.id, with_for_update=True)
        now = await session.scalar(select(func.clock_timestamp()))
        if (
            pair is None
            or task is None
            or task.pair_id != pair.id
            or task.status != "leased"
            or task.fence != lease.fence
            or task.worker_id != lease.worker_id
            or task.lease_until is None
            or task.lease_until <= now
            or task.memory_epoch != pair.memory_epoch
            or lease.memory_epoch != pair.memory_epoch
        ):
            raise DomainError("stale_memory_lease")
        source = await session.get(Message, task.source_message_id)
        if source.deleted or source.revision != task.source_revision:
            raise DomainError("stale_memory_lease")
        return pair, task, source, now

    async def renew(self, lease: MemoryLease, lease_seconds: float):
        if lease_seconds <= 0:
            raise ValueError("Invalid memory lease settings")
        async with self.db.sessions.begin() as session:
            _, task, _, now = await self._lease(session, lease)
            task.lease_until = now + timedelta(seconds=lease_seconds)

    async def fail(
        self, lease: MemoryLease, *, code="extraction_failed", retryable=False, usage=None
    ):
        # Never store exception strings or provider response bodies.
        if code not in {
            "extraction_failed",
            "unsupported_memory",
            "extraction_timeout",
            "provider_failed",
        }:
            code = "extraction_failed"
        async with self.db.sessions.begin() as session:
            try:
                _, task, _, _ = await self._lease(session, lease)
            except DomainError:
                return
            task.status = "pending" if retryable and task.attempts < 3 else "failed"
            task.error_code, task.lease_until, task.worker_id = code, None, None
            if usage is not None:
                task.usage = usage
            task.fence += 1

    async def recover_expired(self, limit=100):
        recovered = 0
        for _ in range(limit):
            async with self.db.sessions.begin() as session:
                now = await session.scalar(select(func.clock_timestamp()))
                pair = await session.scalar(
                    select(Pair)
                    .join(MemoryTask)
                    .where(
                        MemoryTask.status == "leased",
                        MemoryTask.lease_until <= now,
                    )
                    .order_by(MemoryTask.source_order)
                    .limit(1)
                    .with_for_update(of=Pair, skip_locked=True)
                )
                if pair is None:
                    break
                task = await session.scalar(
                    select(MemoryTask)
                    .where(
                        MemoryTask.pair_id == pair.id,
                        MemoryTask.status == "leased",
                    )
                    .with_for_update()
                )
                now = await session.scalar(select(func.clock_timestamp()))
                if task is None or task.lease_until > now:
                    continue
                task.status = (
                    "pending"
                    if task.attempts < 3 and task.memory_epoch == pair.memory_epoch
                    else "failed"
                )
                task.error_code, task.worker_id, task.lease_until = "worker_lost", None, None
                task.fence += 1
                recovered += 1
        return recovered

    async def learn(self, lease: MemoryLease, proposals: list[MemoryProposal], *, usage=None):
        if len(proposals) > 8:
            raise DomainError("too_many_memories", 422)
        async with self.db.sessions.begin() as session:
            try:
                pair, task, source, _ = await self._lease(session, lease)
            except DomainError:
                return LearnResult(status="stale")
            keys = set()
            for proposal in proposals:
                key = (proposal.kind, proposal.predicate, proposal.scope)
                if (
                    any(
                        "\x00" in v
                        for v in (proposal.value, proposal.quote, proposal.time_text or "")
                    )
                    or not proposal.value.strip()
                    or proposal.quote not in source.content
                    or proposal.value not in proposal.quote
                    or proposal.time_text is not None
                    and proposal.time_text not in proposal.quote
                    or not retainable(source.content)
                    or proposal.predicate in FUNCTIONAL
                    and key in keys
                ):
                    raise DomainError("unsupported_memory", 422)
                keys.add(key)
            for proposal in proposals:
                await self._write(session, pair, task, proposal)
            task.status, task.lease_until, task.worker_id = "done", None, None
            task.error_code = None
            task.usage = usage
            return LearnResult(status="committed", written=len(proposals))

    async def _write(self, session, pair, task, proposal):
        source = await session.get(Message, task.source_message_id)
        turn = await session.get(Turn, source.turn_id)
        referenced = normalize_time(proposal.time_text, turn.created_at, turn.source_timezone)
        query = (
            select(MemoryEntry)
            .join(
                MemoryVersion,
                and_(
                    MemoryVersion.memory_id == MemoryEntry.id,
                    MemoryVersion.revision == MemoryEntry.revision,
                ),
            )
            .where(
                MemoryEntry.pair_id == pair.id,
                MemoryEntry.status == "active",
                MemoryEntry.kind == proposal.kind,
                MemoryEntry.predicate == proposal.predicate,
                MemoryEntry.scope == proposal.scope,
            )
        )
        if proposal.predicate not in FUNCTIONAL or proposal.scope == "historical":
            query = query.where(MemoryVersion.value == proposal.value)
        if proposal.kind == "episode":
            query = query.where(
                MemoryVersion.time_text == proposal.time_text,
                MemoryVersion.reference_start.is_not_distinct_from(referenced.start),
                MemoryVersion.reference_end.is_not_distinct_from(referenced.end),
            )
        entry = await session.scalar(query.order_by(MemoryEntry.source_order.desc()).limit(1))
        if entry is not None and entry.source_order > task.source_order:
            # A late job can preserve history, never replace a newer current assertion.
            proposal = proposal.model_copy(update={"scope": "historical"})
            entry = None
        if entry is None:
            entry = MemoryEntry(
                pair_id=pair.id,
                kind=proposal.kind,
                predicate=proposal.predicate,
                scope=proposal.scope,
                source_order=task.source_order,
                revision=1,
            )
            session.add(entry)
            await session.flush()
            version = None
        else:
            version = await session.get(MemoryVersion, (entry.id, entry.revision))
            if (
                version.value != proposal.value
                or version.time_text != proposal.time_text
                or version.reference_start != referenced.start
                or version.reference_end != referenced.end
            ):
                entry.revision += 1
                version = None
            entry.source_order = task.source_order
        if version is None:
            session.add(
                MemoryVersion(
                    memory_id=entry.id,
                    revision=entry.revision,
                    value=proposal.value,
                    time_text=proposal.time_text,
                    reference_start=referenced.start,
                    reference_end=referenced.end,
                    time_precision=referenced.precision,
                    time_basis=referenced.basis,
                    source_timezone=referenced.source_timezone,
                )
            )
            await session.flush()
        key = (entry.id, entry.revision, task.source_message_id)
        if await session.get(MemorySource, key) is None:
            session.add(
                MemorySource(
                    memory_id=entry.id,
                    revision=entry.revision,
                    message_id=task.source_message_id,
                    message_revision=task.source_revision,
                    quote=proposal.quote,
                )
            )
        await session.flush()

    @staticmethod
    async def _view(session, entry, version, *, source_limit=None):
        query = (
            select(MemorySource)
            .where(
                MemorySource.memory_id == entry.id,
                MemorySource.revision == version.revision,
            )
            .order_by(MemorySource.message_id)
        )
        if source_limit is not None:
            query = query.limit(source_limit)
        sources = await session.scalars(query)
        source_views = tuple(SourceView.model_validate(s) for s in sources)
        total = (
            len(source_views)
            if source_limit is None
            else await session.scalar(
                select(func.count())
                .select_from(MemorySource)
                .where(
                    MemorySource.memory_id == entry.id, MemorySource.revision == version.revision
                )
            )
        )
        return MemoryView(
            id=entry.id,
            revision=version.revision,
            kind=entry.kind,
            predicate=entry.predicate,
            scope=entry.scope if entry.revision == version.revision else "historical",
            value=version.value,
            time_text=version.time_text,
            referenced_time=ReferencedTime(
                start=version.reference_start,
                end=version.reference_end,
                precision=version.time_precision,
                basis=version.time_basis,
                source_timezone=version.source_timezone,
            )
            if version.reference_start is not None
            else None,
            recorded_at=version.recorded_at,
            sources=source_views,
            total_sources=total,
        )

    @staticmethod
    async def search(
        session, pair_id, query: str, *, limit=8, reference=None, source_timezone="UTC"
    ):
        """Bounded lexical selection; caller may hold the pair lock for snapshot consistency."""
        tokens = list(dict.fromkeys(tokenize(query)))[:32]
        predicates = [p for p, hints in HINTS.items() if any(h in query.lower() for h in hints)]
        matches = [MemoryEntry.predicate.in_(predicates)] if predicates else []
        matches += [MemoryVersion.value.icontains(token, autoescape=True) for token in tokens]
        if not matches:
            return []
        score = sum(case((condition, 1), else_=0) for condition in matches)
        reference = reference or await session.scalar(select(func.clock_timestamp()))
        windows = query_intervals(query, reference, source_timezone)
        past = PAST.search(query) is not None or bool(windows)
        version_match = MemoryVersion.memory_id == MemoryEntry.id
        scope_match = or_(
            MemoryEntry.scope == "historical", MemoryVersion.revision < MemoryEntry.revision
        )
        time_match = (
            or_(
                *(
                    and_(
                        MemoryVersion.reference_start < window.end,
                        MemoryVersion.reference_end > window.start,
                    )
                    for window in windows
                )
            )
            if windows
            else None
        )
        if windows:
            # A referenced period is not an inferred employment-validity interval.
            scope_match = or_(
                MemoryEntry.kind != "episode", MemoryVersion.reference_start.is_(None), time_match
            )
            score = score + case((time_match, 3), else_=0)
        elif not past:
            version_match = and_(version_match, MemoryVersion.revision == MemoryEntry.revision)
            scope_match = MemoryEntry.scope == "current"
        rows = await session.execute(
            select(MemoryEntry, MemoryVersion)
            .join(
                MemoryVersion,
                version_match,
            )
            .where(
                MemoryEntry.pair_id == pair_id,
                MemoryEntry.status == "active",
                scope_match,
                or_(*matches),
            )
            .order_by(
                score.desc(),
                MemoryEntry.source_order.desc(),
                MemoryVersion.revision.desc(),
                MemoryEntry.id,
            )
            .limit(limit)
        )
        return [await Memory._view(session, entry, version) for entry, version in rows]

    async def recall(self, owner_id: UUID, character_id: UUID, query: str, *, limit=8):
        if not 1 <= limit <= 12 or len(query) > 4000:
            raise DomainError("invalid_memory_query", 422)
        async with self.db.sessions() as session:
            pair_id = await session.scalar(
                select(Pair.id).where(
                    Pair.owner_id == owner_id,
                    Pair.character_id == character_id,
                )
            )
            return (
                []
                if pair_id is None
                else await self.search(
                    session, pair_id, query, limit=limit, source_timezone=self.source_timezone
                )
            )

    @staticmethod
    async def _owned_entry(session, owner_id, memory_id, *, lock=False):
        query = (
            select(Pair)
            .join(MemoryEntry)
            .where(
                Pair.owner_id == owner_id,
                MemoryEntry.id == memory_id,
            )
        )
        pair = await session.scalar(query.with_for_update(of=Pair) if lock else query)
        if pair is None:
            raise DomainError("not_found", 404)
        entry = await session.get(MemoryEntry, memory_id)
        if entry.status != "active":
            raise DomainError("memory_deleted", 410)
        return pair, entry

    async def history(self, owner_id: UUID, memory_id: UUID):
        async with self.db.sessions() as session:
            _, entry = await self._owned_entry(session, owner_id, memory_id)
            versions = await session.scalars(
                select(MemoryVersion)
                .where(
                    MemoryVersion.memory_id == memory_id,
                )
                .order_by(MemoryVersion.revision)
                .limit(101)
            )
            versions = list(versions)
            if len(versions) > 100:
                raise DomainError("pagination_required", 413)
            return [await self._view(session, entry, version) for version in versions]

    async def history_page(self, owner_id, memory_id, *, limit=10, cursor=None):
        async with self.db.sessions() as session:
            _, entry = await self._owned_entry(session, owner_id, memory_id)
            query = select(MemoryVersion).where(MemoryVersion.memory_id == memory_id)
            if cursor is not None:
                query = query.where(MemoryVersion.revision < cursor)
            rows = list(
                await session.scalars(
                    query.order_by(MemoryVersion.revision.desc()).limit(limit + 1)
                )
            )
            return {
                "items": [
                    await self._view(session, entry, v, source_limit=20) for v in rows[:limit]
                ],
                "next_cursor": rows[limit - 1].revision if len(rows) > limit else None,
            }

    async def sources_page(self, owner_id, memory_id, revision, *, limit=20, cursor=None):
        async with self.db.sessions() as session:
            await self._owned_entry(session, owner_id, memory_id)
            if await session.get(MemoryVersion, (memory_id, revision)) is None:
                raise DomainError("not_found", 404)
            query = select(MemorySource).where(
                MemorySource.memory_id == memory_id, MemorySource.revision == revision
            )
            if cursor is not None:
                query = query.where(MemorySource.message_id > cursor)
            rows = list(
                await session.scalars(query.order_by(MemorySource.message_id).limit(limit + 1))
            )
            return {
                "items": [SourceView.model_validate(s) for s in rows[:limit]],
                "next_cursor": str(rows[limit - 1].message_id) if len(rows) > limit else None,
            }

    async def entries(self, owner_id: UUID, character_id: UUID, *, limit=100):
        if not 1 <= limit <= 100:
            raise DomainError("invalid_memory_query", 422)
        async with self.db.sessions() as session:
            rows = await session.execute(
                select(MemoryEntry, MemoryVersion)
                .join(Pair)
                .join(
                    MemoryVersion,
                    and_(
                        MemoryVersion.memory_id == MemoryEntry.id,
                        MemoryVersion.revision == MemoryEntry.revision,
                    ),
                )
                .where(
                    Pair.owner_id == owner_id,
                    Pair.character_id == character_id,
                    MemoryEntry.status == "active",
                )
                .order_by(MemoryEntry.source_order.desc())
                .limit(limit)
            )
            return [await self._view(session, entry, version) for entry, version in rows]

    async def entries_page(self, owner_id, character_id, *, limit=30, cursor=None):
        if not 1 <= limit <= 100:
            raise DomainError("invalid_cursor", 422)
        from sqlalchemy import tuple_

        async with self.db.sessions() as session:
            query = (
                select(MemoryEntry, MemoryVersion)
                .join(Pair)
                .join(
                    MemoryVersion,
                    and_(
                        MemoryVersion.memory_id == MemoryEntry.id,
                        MemoryVersion.revision == MemoryEntry.revision,
                    ),
                )
                .where(
                    Pair.owner_id == owner_id,
                    Pair.character_id == character_id,
                    MemoryEntry.status == "active",
                )
            )
            if cursor:
                pair, anchor = await self._owned_entry(session, owner_id, cursor)
                if pair.character_id != character_id:
                    raise DomainError("invalid_cursor", 422)
                query = query.where(
                    tuple_(MemoryEntry.source_order, MemoryEntry.id)
                    < (anchor.source_order, anchor.id)
                )
            rows = list(
                (
                    await session.execute(
                        query.order_by(
                            MemoryEntry.source_order.desc(), MemoryEntry.id.desc()
                        ).limit(limit + 1)
                    )
                ).all()
            )
            return {
                "items": [
                    await self._view(session, entry, version, source_limit=20)
                    for entry, version in rows[:limit]
                ],
                "next_cursor": str(rows[limit - 1][0].id) if len(rows) > limit else None,
            }

    async def task_status(self, owner_id: UUID, source_message_id: UUID):
        async with self.db.sessions() as session:
            task = await session.scalar(
                select(MemoryTask)
                .join(Pair)
                .where(
                    Pair.owner_id == owner_id,
                    MemoryTask.source_message_id == source_message_id,
                )
            )
            if task is None:
                raise DomainError("not_found", 404)
            return MemoryTaskView.model_validate(task)

    @staticmethod
    async def _closure(session, pair_id, memory_id=None, *, source_message_id=None):
        """Follow recorded dependencies plus literal copies in the owner's raw history.

        Bounds make the first local implementation fail atomically on large graphs.
        Independent later user messages are not descendants of assistant output.
        """
        memories = {memory_id} if memory_id is not None else set()
        messages = {source_message_id} if source_message_id is not None else set()
        turns = set()
        seed = await session.get(Message, source_message_id) if source_message_id else None

        async def ids(query):
            result = set(await session.scalars(query.distinct().limit(2001)))
            if len(result) > 2000:
                raise DomainError("deletion_scope_too_large", 413)
            return result

        while True:
            before = (len(memories), len(messages), len(turns))
            copies = list(
                await session.execute(
                    select(
                        MemoryEntry.kind,
                        MemoryVersion.value,
                        case((MemoryEntry.kind == "episode", MemoryVersion.time_text), else_=None),
                        case(
                            (
                                and_(
                                    MemoryEntry.kind == "episode", MemoryVersion.time_text.is_(None)
                                ),
                                MemorySource.quote,
                            ),
                            else_=None,
                        ),
                    )
                    .select_from(MemoryVersion)
                    .join(MemoryEntry, MemoryEntry.id == MemoryVersion.memory_id)
                    .outerjoin(
                        MemorySource,
                        and_(
                            MemorySource.memory_id == MemoryVersion.memory_id,
                            MemorySource.revision == MemoryVersion.revision,
                        ),
                    )
                    .where(MemoryVersion.memory_id.in_(memories))
                    .distinct()
                    .limit(2001)
                )
            )
            if len(copies) > 2000:
                raise DomainError("deletion_scope_too_large", 413)
            conditions = (
                [Message.content.contains(seed.content, autoescape=True)]
                if seed is not None and seed.content.strip()
                else []
            )
            for kind, value, time_text, quote in copies:
                condition = Message.content.contains(value, autoescape=True)
                if kind == "episode" and (time_text or quote):
                    # A city is not an event identity. Preserve independently dated trips.
                    condition = and_(
                        condition, Message.content.contains(time_text or quote, autoescape=True)
                    )
                conditions.append(condition)
            if conditions:
                # A queued extractor may not yet have attached a repeated statement.
                # Erase exact textual copies too, so recent history cannot restore them.
                messages |= await ids(
                    select(Message.id)
                    .join(Turn)
                    .where(
                        Turn.pair_id == pair_id,
                        Message.deleted.is_(False),
                        or_(*conditions),
                    )
                )
            messages |= await ids(
                select(MemorySource.message_id).where(MemorySource.memory_id.in_(memories))
            )
            if messages:
                memories |= await ids(
                    select(MemorySource.memory_id)
                    .join(MemoryEntry, MemoryEntry.id == MemorySource.memory_id)
                    .where(
                        MemorySource.message_id.in_(messages),
                        MemoryEntry.pair_id == pair_id,
                    )
                )
                turns |= await ids(
                    select(TurnSourceUse.turn_id).where(TurnSourceUse.message_id.in_(messages))
                )
                turns |= await ids(select(Message.turn_id).where(Message.id.in_(messages)))
            turns |= await ids(
                select(TurnMemoryUse.turn_id).where(TurnMemoryUse.memory_id.in_(memories))
            )
            if turns:
                messages |= await ids(
                    select(Message.id).where(
                        Message.turn_id.in_(turns), Message.role == "assistant"
                    )
                )
            if max(len(memories), len(messages), len(turns)) > 2000:
                raise DomainError("deletion_scope_too_large", 413)
            if before == (len(memories), len(messages), len(turns)):
                return memories, messages, turns

    async def preview_forget(self, owner_id: UUID, memory_id: UUID):
        return await self._preview_forget(owner_id, memory_id)

    async def preview_forget_source(self, owner_id: UUID, source_message_id: UUID):
        return await self._preview_forget(owner_id, source_message_id, source=True)

    @staticmethod
    async def _owned_source(session, owner_id, source_id, *, lock=False):
        query = (
            select(Pair)
            .join(Turn, Turn.pair_id == Pair.id)
            .join(Message, Message.turn_id == Turn.id)
            .where(Pair.owner_id == owner_id, Message.id == source_id, Message.role == "user")
        )
        pair = await session.scalar(query.with_for_update(of=Pair) if lock else query)
        if pair is None:
            raise DomainError("not_found", 404)
        message = await session.get(Message, source_id)
        if message.deleted:
            raise DomainError("source_deleted", 410)
        return pair, message

    async def _preview_forget(self, owner_id, target_id, *, source=False):
        async with self.db.sessions.begin() as session:
            lookup = self._owned_source if source else self._owned_entry
            pair, entry = await lookup(session, owner_id, target_id, lock=True)
            memory_id = None if source else target_id
            source_id = target_id if source else None
            memories, messages, turns = await self._closure(
                session, pair.id, memory_id, source_message_id=source_id
            )
            state_count = await session.scalar(
                select(func.count(func.distinct(DialogueSource.conversation_id))).where(
                    DialogueSource.message_id.in_(messages)
                )
            )
            return ForgetPreview(
                memory_id=memory_id,
                source_message_id=source_id,
                revision=entry.revision,
                memories=len(memories),
                messages=len(messages),
                turns=len(turns),
                dialogue_states=state_count,
                framework_spaces=await framework_space_count(session, pair.id),
            )

    async def deletion(self, owner_id: UUID, deletion_id: UUID):
        async with self.db.sessions() as session:
            receipt = await session.scalar(
                select(MemoryDeletion)
                .join(Pair)
                .where(MemoryDeletion.id == deletion_id, Pair.owner_id == owner_id)
            )
            if receipt is None:
                raise DomainError("not_found", 404)
            return DeletionReceipt.model_validate(receipt)

    async def forget(
        self, owner_id: UUID, memory_id: UUID, request_id: str, *, expected_revision: int
    ):
        return await self._forget(
            owner_id, memory_id, request_id, expected_revision=expected_revision
        )

    async def forget_source(
        self, owner_id: UUID, source_message_id: UUID, request_id: str, *, expected_revision: int
    ):
        return await self._forget(
            owner_id,
            source_message_id,
            request_id,
            expected_revision=expected_revision,
            source=True,
        )

    async def _forget(self, owner_id, target_id, request_id, *, expected_revision, source=False):
        operation = "source.forget" if source else "memory.forget"
        digest = self.ledger.digest(
            str(target_id), {"revision": expected_revision, "action": "forget"}
        )
        async with self.db.sessions.begin() as session:
            await self.ledger.owner_lock(session, owner_id)
            existing = await self.ledger.request(session, owner_id, operation, request_id, digest)
            if existing:
                return DeletionReceipt.model_validate(await session.get(MemoryDeletion, existing))
            lookup = self._owned_source if source else self._owned_entry
            pair, entry = await lookup(session, owner_id, target_id, lock=True)
            if entry.revision != expected_revision:
                raise DomainError(
                    "source_revision_conflict" if source else "memory_revision_conflict"
                )
            memory_id = None if source else target_id
            source_id = target_id if source else None
            memories, messages, turns = await self._closure(
                session, pair.id, memory_id, source_message_id=source_id
            )
            pair.memory_epoch += 1
            from hmph.companion import forget_controls
            from hmph.models import CompanionMention, RepairState

            forget_controls(pair, memories)
            await session.execute(
                delete(RepairState).where(
                    or_(
                        RepairState.source_message_id.in_(messages),
                        RepairState.source_memory_id.in_(memories),
                    )
                )
            )
            await session.execute(
                delete(CompanionMention).where(
                    or_(
                        CompanionMention.source_message_id.in_(messages),
                        CompanionMention.turn_id.in_(turns),
                    )
                )
            )
            active = await session.scalars(
                select(Turn)
                .where(
                    Turn.pair_id == pair.id,
                    Turn.status.in_(ACTIVE),
                )
                .with_for_update()
            )
            for turn in active:
                await terminate_turn(session, turn, "invalidated", "memory_changed")
            # Old extraction snapshots are discarded, including unrelated pending work.
            # This conservative barrier prevents a stale extraction from resurrecting data.
            await session.execute(
                update(MemoryTask)
                .where(
                    MemoryTask.pair_id == pair.id,
                    MemoryTask.status.in_(("pending", "leased")),
                )
                .values(
                    status="done",
                    fence=MemoryTask.fence + 1,
                    worker_id=None,
                    lease_until=None,
                    error_code="memory_epoch_changed",
                )
            )
            await session.execute(delete(MemorySource).where(MemorySource.memory_id.in_(memories)))
            await session.execute(
                delete(MemoryVersion).where(MemoryVersion.memory_id.in_(memories))
            )
            await session.execute(
                update(MemoryEntry)
                .where(MemoryEntry.id.in_(memories))
                .values(
                    status="deleted",
                    predicate="deleted",
                )
            )
            if messages:
                await session.execute(
                    update(Message)
                    .where(Message.id.in_(messages), Message.deleted.is_(False))
                    .values(
                        content="[已删除]",
                        deleted=True,
                        revision=Message.revision + 1,
                    )
                )
            if turns:
                await session.execute(
                    delete(StreamEvent).where(
                        StreamEvent.turn_id.in_(turns),
                        StreamEvent.type == "text.delta",
                    )
                )
                affected = await session.scalars(
                    select(Turn).where(Turn.id.in_(turns)).with_for_update()
                )
                for turn in affected:
                    turn.content_redacted = True
                    turn.companion_plan = None
                    append_event(
                        session, turn, "turn.content_redacted", {"reason": "memory_forgotten"}
                    )
            state_count = await invalidate_dialogue(session, pair, messages)
            receipt = MemoryDeletion(
                pair_id=pair.id,
                memory_id=memory_id,
                source_message_id=source_id,
                memories=len(memories),
                messages=len(messages),
                turns=len(turns),
                status="completed",
                dialogue_states=state_count,
            )
            session.add(receipt)
            await session.flush()
            await retire_for_forget(session, pair, receipt)
            await session.flush()
            self.ledger.save(session, owner_id, operation, request_id, digest, receipt.id)
            return DeletionReceipt.model_validate(receipt)
