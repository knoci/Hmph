"""Budgeted coalescing for the existing shadow-build queue; never selects a backend."""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import func, select

from hmph.contracts import DomainError
from hmph.models import MemoryBuild, MemoryDeletion, Pair


class MaintenancePolicy(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    revision: int = 0
    enabled: bool = False
    backend: Literal["mem0", "graphiti"] = "graphiti"
    model: Literal["deepseek-flash", "deepseek-v4-pro"] = "deepseek-flash"
    cooldown_seconds: int = Field(default=300, ge=1, le=86400)
    minimum_sources: int = Field(default=8, ge=1, le=100)
    per_attempt_calls: int = Field(default=64, ge=1, le=128)
    granted_calls: int = Field(default=0, ge=0)
    remaining_calls: int = Field(default=0, ge=0)
    reserved_calls: int = Field(default=0, ge=0)
    last_scheduled_at: datetime | None = None
    suspended_reason: str | None = None


async def due_policy(session, pair, revision, backend, model):
    policy = MaintenancePolicy.model_validate(pair.memory_maintenance)
    if (
        not policy.enabled
        or policy.revision != revision
        or policy.backend != backend
        or policy.model != model
        or policy.remaining_calls <= 0
    ):
        return None
    now = await session.scalar(select(func.clock_timestamp()))
    if (
        policy.last_scheduled_at is not None
        and (now - policy.last_scheduled_at).total_seconds() < policy.cooldown_seconds
    ):
        return None
    active = await session.scalar(
        select(MemoryBuild.id)
        .where(
            MemoryBuild.pair_id == pair.id,
            MemoryBuild.backend == backend,
            MemoryBuild.status.in_(("queued", "building")),
        )
        .limit(1)
    )
    pending_forget = await session.scalar(
        select(MemoryDeletion.id)
        .where(
            MemoryDeletion.pair_id == pair.id,
            MemoryDeletion.status == "pending_cleanup",
        )
        .limit(1)
    )
    return None if active or pending_forget else policy


def suspend_policy(pair, revision, reason):
    policy = MaintenancePolicy.model_validate(pair.memory_maintenance)
    if policy.revision == revision:
        pair.memory_maintenance = policy.model_copy(
            update={
                "enabled": False,
                "suspended_reason": reason,
            }
        ).model_dump(mode="json")


class MemoryMaintenance:
    def __init__(self, builds):
        self.builds, self.db, self.ledger = builds, builds.db, builds.ledger

    async def read(self, owner_id, character_id):
        async with self.db.sessions() as session:
            pair = await self.builds._pair(session, owner_id, character_id)
            return MaintenancePolicy.model_validate(pair.memory_maintenance)

    async def update(
        self,
        owner_id,
        character_id,
        request_id,
        *,
        expected_revision,
        enabled,
        backend="graphiti",
        model="deepseek-flash",
        cooldown_seconds=300,
        minimum_sources=8,
        per_attempt_calls=64,
        additional_calls=0,
    ):
        if type(additional_calls) is not int or not 0 <= additional_calls <= 4096:
            raise DomainError("invalid_maintenance_budget", 422)
        settings = MaintenancePolicy(
            enabled=enabled,
            backend=backend,
            model=model,
            cooldown_seconds=cooldown_seconds,
            minimum_sources=minimum_sources,
            per_attempt_calls=per_attempt_calls,
        )
        body = {
            **settings.model_dump(mode="json"),
            "expected_revision": expected_revision,
            "additional_calls": additional_calls,
        }
        digest = self.ledger.digest(str(character_id), body)
        async with self.db.sessions.begin() as session:
            await self.ledger.owner_lock(session, owner_id)
            existing = await self.ledger.request(
                session, owner_id, "memory.maintenance", request_id, digest
            )
            pair = await self.builds._pair(session, owner_id, character_id, lock=True)
            previous = MaintenancePolicy.model_validate(pair.memory_maintenance)
            if existing:
                return previous
            if previous.revision != expected_revision:
                raise DomainError("maintenance_revision_conflict")
            remaining = previous.remaining_calls + additional_calls
            if enabled and remaining <= 0:
                raise DomainError("maintenance_budget_required", 422)
            # An explicit owner update invalidates only maintenance queue/writers.
            active = list(
                await session.scalars(
                    select(MemoryBuild.id).where(
                        MemoryBuild.pair_id == pair.id,
                        MemoryBuild.maintenance_revision.is_not(None),
                        MemoryBuild.status.in_(("queued", "building")),
                    )
                )
            )
            await self.builds._retire(session, active)
            policy = settings.model_copy(
                update={
                    "revision": previous.revision + 1,
                    "granted_calls": previous.granted_calls + additional_calls,
                    "remaining_calls": remaining,
                    "reserved_calls": previous.reserved_calls,
                    "last_scheduled_at": previous.last_scheduled_at,
                }
            )
            pair.memory_maintenance = policy.model_dump(mode="json")
            self.ledger.save(session, owner_id, "memory.maintenance", request_id, digest, pair.id)
            return policy

    async def schedule(self, backend, *, limit=20):
        async with self.db.sessions() as session:
            pairs = list(
                (
                    await session.execute(
                        select(
                            Pair.owner_id,
                            Pair.character_id,
                            Pair.memory_maintenance,
                            Pair.next_memory_seq,
                            Pair.memory_epoch,
                        )
                        .where(
                            Pair.memory_maintenance["enabled"].as_boolean().is_(True),
                            Pair.memory_maintenance["backend"].as_string() == backend,
                        )
                        .order_by(Pair.id)
                        .limit(min(20, max(1, limit)))
                    )
                ).all()
            )
        queued = []
        for owner, character, data, next_source, epoch in pairs:
            policy = MaintenancePolicy.model_validate(data)
            try:
                result = await self.builds.request(
                    owner,
                    character,
                    f"maintenance:{policy.revision}:{epoch}:{next_source - 1}:{character}",
                    backend=backend,
                    model=policy.model,
                    maintenance_revision=policy.revision,
                )
                if result is not None and result.status == "queued":
                    queued.append(result.id)
            except DomainError as error:
                if error.code not in {"framework_snapshot_too_large"}:
                    raise
                async with self.db.sessions.begin() as session:
                    pair = await self.builds._pair(session, owner, character, lock=True)
                    suspend_policy(pair, policy.revision, "snapshot_too_large")
        return queued
