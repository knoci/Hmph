"""Durable shadow builds. Pair locks serialize source snapshots, publish and forget.

Every writer attempt owns a new space. Lease expiry never proves writer death;
only an explicit quiescence acknowledgement makes a retired space cleanable.
No native framework or network operation runs inside these transactions.
"""

from datetime import timedelta

from sqlalchemy import delete, func, select, update

from hmph.contracts import DomainError
from hmph.graphiti_reference import GRAPHITI_REVISION
from hmph.mem0_reference import MEM0_REVISION
from hmph.memory_build_types import (
    BuildLease,
    BuildSource,
    BuildUsage,
    BuildView,
    CleanupLease,
    SpaceView,
    WriterIdentity,
)
from hmph.models import (
    MemoryBuild,
    MemoryBuildSource,
    MemoryDeletion,
    MemoryDeletionSpace,
    MemorySpace,
    MemoryTask,
    Message,
    Pair,
    Turn,
)
from hmph.persistence import RequestLedger
from hmph.source_policy import retainable

REVISIONS = {"mem0": MEM0_REVISION, "graphiti": GRAPHITI_REVISION}
MODELS = {"deepseek-flash", "deepseek-v4-pro"}
MAX_SOURCES = 1000
MAX_SOURCE_BYTES = 2 * 1024 * 1024


async def framework_space_count(session, pair_id):
    return await session.scalar(
        select(func.count())
        .select_from(MemorySpace)
        .join(MemoryBuild)
        .where(MemoryBuild.pair_id == pair_id, MemorySpace.status != "cleaned")
    )


async def retire_for_forget(session, pair, receipt):
    """Runs in the authoritative forget transaction, while its pair lock is held."""
    build_ids = list(
        await session.scalars(
            select(MemoryBuild.id).where(
                MemoryBuild.pair_id == pair.id, MemoryBuild.status != "retired"
            )
        )
    )
    await MemoryBuilds._retire(session, build_ids)
    all_builds = select(MemoryBuild.id).where(MemoryBuild.pair_id == pair.id)
    await session.execute(
        delete(MemoryBuildSource).where(MemoryBuildSource.build_id.in_(all_builds))
    )
    spaces = list(
        await session.scalars(
            select(MemorySpace.id)
            .join(MemoryBuild)
            .where(MemoryBuild.pair_id == pair.id, MemorySpace.status != "cleaned")
            .limit(2001)
        )
    )
    if len(spaces) > 2000:
        raise DomainError("deletion_scope_too_large", 413)
    receipt.framework_spaces = len(spaces)
    if spaces:
        receipt.status, receipt.completed_at = "pending_cleanup", None
        session.add_all([MemoryDeletionSpace(deletion_id=receipt.id, space_id=s) for s in spaces])


class MemoryBuilds:
    def __init__(self, db, *, digest_key):
        self.db = db
        self.ledger = RequestLedger(digest_key)

    async def _pair(self, session, owner_id, character_id, *, lock=False):
        query = select(Pair).where(Pair.owner_id == owner_id, Pair.character_id == character_id)
        pair = await session.scalar(query.with_for_update() if lock else query)
        if pair is None:
            raise DomainError("not_found", 404)
        return pair

    async def request(self, owner_id, character_id, request_id, *, backend, model="deepseek-flash"):
        if backend not in REVISIONS or model not in MODELS:
            raise DomainError("invalid_framework_config", 422)
        digest = self.ledger.digest(str(character_id), {"backend": backend, "model": model})
        async with self.db.sessions.begin() as session:
            await self.ledger.owner_lock(session, owner_id)
            existing = await self.ledger.request(
                session, owner_id, "memory.build", request_id, digest
            )
            if existing:
                return BuildView.model_validate(await session.get(MemoryBuild, existing))
            pair = await self._pair(session, owner_id, character_id, lock=True)
            rows = (
                await session.execute(
                    select(Message, MemoryTask.source_order)
                    .join(MemoryTask, MemoryTask.source_message_id == Message.id)
                    .where(
                        MemoryTask.pair_id == pair.id,
                        MemoryTask.source_revision == Message.revision,
                        Message.role == "user",
                        Message.deleted.is_(False),
                    )
                    .order_by(MemoryTask.source_order)
                    .limit(MAX_SOURCES + 1)
                )
            ).all()
            if len(rows) > MAX_SOURCES:
                raise DomainError("framework_snapshot_too_large", 413)
            rows = [(m, order) for m, order in rows if retainable(m.content)]
            if sum(len(m.content.encode()) for m, _ in rows) > MAX_SOURCE_BYTES:
                raise DomainError("framework_snapshot_too_large", 413)
            # Supersede unfinished builds. A previously ready space remains until publication.
            obsolete = list(
                await session.scalars(
                    select(MemoryBuild.id).where(
                        MemoryBuild.pair_id == pair.id,
                        MemoryBuild.backend == backend,
                        MemoryBuild.status.in_(("queued", "building")),
                    )
                )
            )
            await self._retire(session, obsolete)
            build = MemoryBuild(
                pair_id=pair.id,
                generation=pair.next_framework_seq,
                memory_epoch=pair.memory_epoch,
                backend=backend,
                backend_revision=REVISIONS[backend],
                model=model,
                watermark=pair.next_memory_seq - 1,
                source_count=len(rows),
                snapshot_digest=self.ledger.digest(
                    str(pair.id), [[str(m.id), m.revision, order] for m, order in rows]
                ),
            )
            pair.next_framework_seq += 1
            session.add(build)
            await session.flush()
            session.add_all(
                [
                    MemoryBuildSource(
                        build_id=build.id,
                        message_id=m.id,
                        message_revision=m.revision,
                        source_order=order,
                    )
                    for m, order in rows
                ]
            )
            self.ledger.save(session, owner_id, "memory.build", request_id, digest, build.id)
            return BuildView.model_validate(build)

    @staticmethod
    async def _retire(session, ids):
        if not ids:
            return
        await session.execute(
            update(MemoryBuild)
            .where(MemoryBuild.id.in_(ids))
            .values(status="retired", fence=MemoryBuild.fence + 1, worker_id=None, lease_until=None)
        )
        await session.execute(
            update(MemorySpace)
            .where(MemorySpace.build_id.in_(ids), MemorySpace.status.in_(("writing", "ready")))
            .values(status="retired")
        )

    async def get(self, owner_id, build_id):
        async with self.db.sessions() as session:
            build = await session.scalar(
                select(MemoryBuild)
                .join(Pair)
                .where(MemoryBuild.id == build_id, Pair.owner_id == owner_id)
            )
            if build is None:
                raise DomainError("not_found", 404)
            return BuildView.model_validate(build)

    async def current(self, owner_id, character_id, *, backend):
        async with self.db.sessions() as session:
            pair = await self._pair(session, owner_id, character_id)
            build = await session.scalar(
                select(MemoryBuild).where(
                    MemoryBuild.pair_id == pair.id,
                    MemoryBuild.backend == backend,
                    MemoryBuild.memory_epoch == pair.memory_epoch,
                    MemoryBuild.status == "ready",
                )
            )
            return BuildView.model_validate(build) if build else None

    async def spaces(self, owner_id, build_id):
        async with self.db.sessions() as session:
            build = await session.scalar(
                select(MemoryBuild)
                .join(Pair)
                .where(MemoryBuild.id == build_id, Pair.owner_id == owner_id)
            )
            if build is None:
                raise DomainError("not_found", 404)
            spaces = await session.scalars(
                select(MemorySpace)
                .where(MemorySpace.build_id == build_id)
                .order_by(MemorySpace.fence)
            )
            return [SpaceView.model_validate(s) for s in spaces]

    @staticmethod
    async def _sources(session, build):
        rows = (
            await session.execute(
                select(MemoryBuildSource, Message, Turn)
                .join(Message, MemoryBuildSource.message_id == Message.id)
                .join(Turn, Message.turn_id == Turn.id)
                .where(MemoryBuildSource.build_id == build.id)
                .order_by(MemoryBuildSource.source_order)
            )
        ).all()
        if len(rows) != build.source_count or any(
            m.deleted
            or m.revision != source.message_revision
            or m.role != "user"
            or turn.pair_id != build.pair_id
            or not retainable(m.content)
            for source, m, turn in rows
        ):
            raise DomainError("framework_source_changed")
        return tuple(
            BuildSource(
                message_id=m.id,
                revision=m.revision,
                source_order=source.source_order,
                content=m.content,
                observed_at=turn.created_at,
                source_timezone=turn.source_timezone,
            )
            for source, m, turn in rows
        )

    async def claim(self, worker_id, *, backend, lease_seconds=30):
        self._validate_worker(worker_id, lease_seconds)
        async with self.db.sessions.begin() as session:
            pair = await session.scalar(
                select(Pair)
                .join(MemoryBuild)
                .where(
                    MemoryBuild.status == "queued",
                    MemoryBuild.backend == backend,
                )
                .order_by(MemoryBuild.created_at, MemoryBuild.id)
                .limit(1)
                .with_for_update(of=Pair, skip_locked=True)
            )
            if pair is None:
                return None
            build = await session.scalar(
                select(MemoryBuild)
                .where(
                    MemoryBuild.pair_id == pair.id,
                    MemoryBuild.backend == backend,
                    MemoryBuild.status == "queued",
                )
                .order_by(MemoryBuild.generation.desc())
                .limit(1)
                .with_for_update()
            )
            if build is None:
                return None
            if build.memory_epoch != pair.memory_epoch:
                await self._retire(session, [build.id])
                return None
            try:
                sources = await self._sources(session, build)
            except DomainError:
                build.status, build.error_code = "failed", "framework_source_changed"
                return None
            now = await session.scalar(select(func.clock_timestamp()))
            build.status, build.worker_id = "building", worker_id
            build.error_code = None
            build.fence += 1
            build.attempts += 1
            build.lease_until = now + timedelta(seconds=lease_seconds)
            space = MemorySpace(build_id=build.id, fence=build.fence, worker_id=worker_id)
            session.add(space)
            await session.flush()
            return BuildLease(
                build_id=build.id,
                space_id=space.id,
                pair_id=pair.id,
                memory_epoch=build.memory_epoch,
                backend=build.backend,
                backend_revision=build.backend_revision,
                model=build.model,
                fence=build.fence,
                worker_id=worker_id,
                sources=sources,
            )

    @staticmethod
    def _validate_worker(worker_id, lease_seconds):
        if not 1 <= len(worker_id) <= 100 or not 0 < lease_seconds <= 300:
            raise ValueError("Invalid framework lease settings")

    async def _lease(self, session, lease):
        pair = await session.scalar(select(Pair).where(Pair.id == lease.pair_id).with_for_update())
        build = await session.get(MemoryBuild, lease.build_id, with_for_update=True)
        space = await session.get(MemorySpace, lease.space_id, with_for_update=True)
        now = await session.scalar(select(func.clock_timestamp()))
        if (
            pair is None
            or build is None
            or space is None
            or build.pair_id != pair.id
            or space.build_id != build.id
            or build.memory_epoch != pair.memory_epoch
            or lease.memory_epoch != pair.memory_epoch
            or build.status != "building"
            or space.status != "writing"
            or build.fence != lease.fence
            or space.fence != lease.fence
            or build.worker_id != lease.worker_id
            or space.worker_id != lease.worker_id
            or build.backend != lease.backend
            or build.backend_revision != lease.backend_revision
            or build.model != lease.model
            or build.lease_until is None
            or build.lease_until <= now
        ):
            raise DomainError("stale_framework_lease")
        return pair, build, space, now

    async def quiesce(self, lease, *, usage=None):
        """Trusted writer calls only after ALL native writes and child work have stopped."""
        async with self.db.sessions.begin() as session:
            await session.scalar(select(Pair).where(Pair.id == lease.pair_id).with_for_update())
            space = await session.scalar(
                select(MemorySpace)
                .join(MemoryBuild)
                .where(MemorySpace.id == lease.space_id, MemoryBuild.pair_id == lease.pair_id)
                .with_for_update(of=MemorySpace)
            )
            if (
                space is None
                or space.build_id != lease.build_id
                or space.fence != lease.fence
                or space.worker_id != lease.worker_id
            ):
                raise DomainError("stale_framework_lease")
            if space.execution is not None:
                raise DomainError("framework_stop_proof_required")
            space.quiesced = True
            if usage is not None:
                space.usage = BuildUsage.model_validate(usage).model_dump()

    async def reserve_execution(self, lease, identity):
        async with self.db.sessions.begin() as session:
            _, _, space, _ = await self._lease(session, lease)
            if space.execution is not None or space.quiesced or identity.pid is not None:
                raise DomainError("framework_execution_conflict")
            space.execution = identity.model_dump(mode="json")

    async def start_execution(self, lease, reserved, running):
        async with self.db.sessions.begin() as session:
            _, _, space, _ = await self._lease(session, lease)
            if (
                space.quiesced
                or space.execution != reserved.model_dump(mode="json")
                or running.model_copy(update={"pid": None, "started_at": None}) != reserved
                or running.pid is None
                or running.started_at is None
            ):
                raise DomainError("framework_execution_conflict")
            space.execution = running.model_dump(mode="json")

    async def execution(self, lease):
        async with self.db.sessions() as session:
            space = await session.get(MemorySpace, lease.space_id)
            if (
                space is None
                or space.build_id != lease.build_id
                or space.fence != lease.fence
                or space.worker_id != lease.worker_id
            ):
                raise DomainError("stale_framework_lease")
            return WriterIdentity.model_validate(space.execution) if space.execution else None

    async def validate_execution(self, lease, running):
        async with self.db.sessions.begin() as session:
            _, _, space, _ = await self._lease(session, lease)
            if space.quiesced or space.execution != running.model_dump(mode="json"):
                raise DomainError("stale_framework_lease")

    async def recoverable_executions(self, *, backend, host_id=None):
        async with self.db.sessions() as session:
            query = (
                select(MemorySpace, MemoryBuild)
                .join(MemoryBuild)
                .where(
                    MemoryBuild.backend == backend,
                    MemorySpace.status == "retired",
                    MemorySpace.quiesced.is_(False),
                    MemorySpace.execution.is_not(None),
                )
            )
            if host_id is not None:
                query = query.where(MemorySpace.execution["host_id"].astext == host_id)
            rows = (await session.execute(query.order_by(MemorySpace.id).limit(100))).all()
            return [
                (
                    BuildLease(
                        build_id=build.id,
                        space_id=space.id,
                        pair_id=build.pair_id,
                        memory_epoch=build.memory_epoch,
                        backend=build.backend,
                        backend_revision=build.backend_revision,
                        model=build.model,
                        fence=space.fence,
                        worker_id=space.worker_id,
                        sources=(),
                    ),
                    WriterIdentity.model_validate(space.execution),
                )
                for space, build in rows
            ]

    async def confirm_execution_stopped(self, lease, expected, *, usage=None):
        """Trusted supervisor holds the OS lock and has drained backend operations."""
        async with self.db.sessions.begin() as session:
            await session.scalar(select(Pair).where(Pair.id == lease.pair_id).with_for_update())
            space = await session.scalar(
                select(MemorySpace)
                .join(MemoryBuild)
                .where(
                    MemorySpace.id == lease.space_id,
                    MemoryBuild.pair_id == lease.pair_id,
                )
                .with_for_update(of=MemorySpace)
            )
            if (
                space is None
                or space.build_id != lease.build_id
                or space.fence != lease.fence
                or space.worker_id != lease.worker_id
                or space.execution != expected.model_dump(mode="json")
            ):
                raise DomainError("stale_framework_lease")
            space.quiesced = True
            space.stop_proof = "process_lock_and_database_barrier"
            if usage is not None:
                space.usage = BuildUsage.model_validate(usage).model_dump()

    async def stop_unverified(self, lease):
        async with self.db.sessions.begin() as session:
            await session.scalar(select(Pair).where(Pair.id == lease.pair_id).with_for_update())
            await session.execute(
                update(MemorySpace)
                .where(
                    MemorySpace.id == lease.space_id,
                    MemorySpace.build_id == lease.build_id,
                    MemorySpace.fence == lease.fence,
                    MemorySpace.worker_id == lease.worker_id,
                    MemorySpace.quiesced.is_(False),
                )
                .values(error_code="framework_stop_unverified")
            )

    async def publish(self, lease):
        async with self.db.sessions.begin() as session:
            try:
                pair, build, space, now = await self._lease(session, lease)
            except DomainError:
                return False
            if not space.quiesced:
                raise DomainError("framework_writer_not_quiesced")
            await self._sources(session, build)
            previous = list(
                await session.scalars(
                    select(MemoryBuild.id).where(
                        MemoryBuild.pair_id == pair.id,
                        MemoryBuild.backend == build.backend,
                        MemoryBuild.status == "ready",
                    )
                )
            )
            await self._retire(session, previous)
            build.status, build.ready_at = "ready", now
            build.worker_id, build.lease_until = None, None
            space.status = "ready"
            return True

    async def renew(self, lease, lease_seconds):
        self._validate_worker(lease.worker_id, lease_seconds)
        async with self.db.sessions.begin() as session:
            _, build, _, now = await self._lease(session, lease)
            build.lease_until = now + timedelta(seconds=lease_seconds)

    async def fail(self, lease, *, code="build_failed", retryable=False):
        if code not in {"build_failed", "provider_failed", "call_budget", "lease_lost"}:
            raise ValueError("Invalid framework failure code")
        async with self.db.sessions.begin() as session:
            try:
                _, build, space, _ = await self._lease(session, lease)
            except DomainError:
                return False
            space.status = "retired"
            space.error_code = code
            build.status = "queued" if retryable and build.attempts < 3 else "failed"
            build.worker_id, build.lease_until = None, None
            build.error_code = code
            return True

    async def recover_expired(self, *, backend):
        async with self.db.sessions.begin() as session:
            now = await session.scalar(select(func.clock_timestamp()))
            pairs = list(
                await session.scalars(
                    select(Pair)
                    .where(
                        Pair.id.in_(
                            select(MemoryBuild.pair_id).where(
                                MemoryBuild.backend == backend,
                                MemoryBuild.status == "building",
                                MemoryBuild.lease_until <= now,
                            )
                        )
                    )
                    .order_by(Pair.id)
                    .limit(100)
                    .with_for_update(skip_locked=True)
                )
            )
            for pair in pairs:
                builds = list(
                    await session.scalars(
                        select(MemoryBuild)
                        .where(
                            MemoryBuild.pair_id == pair.id,
                            MemoryBuild.backend == backend,
                            MemoryBuild.status == "building",
                            MemoryBuild.lease_until <= now,
                        )
                        .with_for_update()
                    )
                )
                for build in builds:
                    await session.execute(
                        update(MemorySpace)
                        .where(MemorySpace.build_id == build.id, MemorySpace.status == "writing")
                        .values(status="retired", error_code="worker_lost")
                    )
                    build.status = "queued" if build.attempts < 3 else "failed"
                    build.fence += 1
                    build.error_code = "worker_lost"
                    build.worker_id, build.lease_until = None, None
            # Retrying cleanup is safe only for spaces already acknowledged quiescent.
            await session.execute(
                update(MemorySpace)
                .where(
                    MemorySpace.build_id.in_(
                        select(MemoryBuild.id).where(MemoryBuild.backend == backend)
                    ),
                    MemorySpace.status == "cleaning",
                    MemorySpace.cleanup_until <= now,
                )
                .values(status="retired", cleanup_worker_id=None, cleanup_until=None)
            )

    async def claim_cleanup(self, worker_id, *, backend, lease_seconds=30):
        self._validate_worker(worker_id, lease_seconds)
        async with self.db.sessions.begin() as session:
            now = await session.scalar(select(func.clock_timestamp()))
            eligible = (
                select(MemoryBuild.pair_id)
                .join(MemorySpace)
                .where(
                    MemoryBuild.backend == backend,
                    MemorySpace.status == "retired",
                    MemorySpace.quiesced.is_(True),
                    (MemorySpace.cleanup_until.is_(None) | (MemorySpace.cleanup_until <= now)),
                )
            )
            pair = await session.scalar(
                select(Pair)
                .where(Pair.id.in_(eligible))
                .order_by(Pair.id)
                .limit(1)
                .with_for_update(skip_locked=True)
            )
            if pair is None:
                return None
            row = (
                await session.execute(
                    select(MemorySpace, MemoryBuild.backend)
                    .join(MemoryBuild)
                    .where(
                        MemoryBuild.pair_id == pair.id,
                        MemoryBuild.backend == backend,
                        MemorySpace.status == "retired",
                        MemorySpace.quiesced.is_(True),
                        (MemorySpace.cleanup_until.is_(None) | (MemorySpace.cleanup_until <= now)),
                    )
                    .order_by(MemorySpace.id)
                    .limit(1)
                    .with_for_update(of=MemorySpace)
                )
            ).first()
            if row is None:
                return None
            space, backend = row
            space.status, space.cleanup_worker_id = "cleaning", worker_id
            space.cleanup_fence += 1
            space.cleanup_attempts += 1
            space.cleanup_until = now + timedelta(seconds=lease_seconds)
            return CleanupLease(
                space_id=space.id,
                pair_id=pair.id,
                backend=backend,
                worker_id=worker_id,
                fence=space.cleanup_fence,
            )

    async def _cleanup_lease(self, session, lease):
        await session.scalar(select(Pair).where(Pair.id == lease.pair_id).with_for_update())
        row = (
            await session.execute(
                select(MemorySpace, MemoryBuild)
                .join(MemoryBuild)
                .where(
                    MemorySpace.id == lease.space_id,
                    MemoryBuild.pair_id == lease.pair_id,
                    MemoryBuild.backend == lease.backend,
                )
                .with_for_update(of=MemorySpace)
            )
        ).first()
        now = await session.scalar(select(func.clock_timestamp()))
        if row is None:
            return None, now
        space, _ = row
        if (
            space.status != "cleaning"
            or not space.quiesced
            or space.cleanup_fence != lease.fence
            or space.cleanup_worker_id != lease.worker_id
            or space.cleanup_until is None
            or space.cleanup_until <= now
        ):
            return None, now
        return space, now

    async def finish_cleanup(self, lease, proof):
        if not (proof.store_empty and proof.auxiliary_empty):
            raise DomainError("framework_cleanup_unverified")
        async with self.db.sessions.begin() as session:
            space, now = await self._cleanup_lease(session, lease)
            if space is None:
                return False
            space.status, space.cleaned_at = "cleaned", now
            space.cleanup_worker_id, space.cleanup_until, space.error_code = None, None, None
            await session.flush()
            receipts = list(
                await session.scalars(
                    select(MemoryDeletion)
                    .where(
                        MemoryDeletion.id.in_(
                            select(MemoryDeletionSpace.deletion_id).where(
                                MemoryDeletionSpace.space_id == space.id
                            )
                        ),
                        MemoryDeletion.status == "pending_cleanup",
                    )
                    .with_for_update()
                )
            )
            for receipt in receipts:
                remaining = await session.scalar(
                    select(func.count())
                    .select_from(MemoryDeletionSpace)
                    .join(MemorySpace)
                    .where(
                        MemoryDeletionSpace.deletion_id == receipt.id,
                        MemorySpace.status != "cleaned",
                    )
                )
                if not remaining:
                    receipt.status, receipt.completed_at = "completed", now
            return True

    async def fail_cleanup(self, lease, *, retry_seconds=5):
        if not 0 <= retry_seconds <= 300:
            raise ValueError("Invalid cleanup retry delay")
        async with self.db.sessions.begin() as session:
            space, now = await self._cleanup_lease(session, lease)
            if space is None:
                return False
            space.status, space.error_code = "retired", "framework_cleanup_failed"
            space.cleanup_worker_id = None
            space.cleanup_until = now + timedelta(seconds=retry_seconds)
            return True
