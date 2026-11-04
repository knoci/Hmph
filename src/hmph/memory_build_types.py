"""Framework orchestration contracts; manifests reference authoritative messages."""

from datetime import datetime
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class FrameworkBuildError(Exception):
    def __init__(self, code, *, retryable):
        if code not in {"build_failed", "provider_failed", "call_budget"}:
            raise ValueError("Invalid framework error code")
        self.code, self.retryable = code, retryable
        super().__init__(code)


class BuildView(BaseModel):
    model_config = ConfigDict(from_attributes=True, frozen=True)
    id: UUID
    backend: str
    backend_revision: str
    model: str
    generation: int
    memory_epoch: int
    status: str
    watermark: int
    source_count: int
    snapshot_digest: str
    attempts: int
    maintenance_revision: int | None = None
    call_limit: int | None = None
    error_code: str | None
    created_at: datetime
    ready_at: datetime | None
    mode: Literal["shadow"] = "shadow"


class BuildSource(BaseModel):
    model_config = ConfigDict(frozen=True)
    message_id: UUID
    revision: int
    source_order: int
    content: str
    observed_at: datetime
    source_timezone: str | None


class BuildLease(BaseModel):
    model_config = ConfigDict(frozen=True)
    build_id: UUID
    space_id: UUID
    pair_id: UUID
    memory_epoch: int
    backend: str
    backend_revision: str
    model: str
    fence: int
    worker_id: str
    sources: tuple[BuildSource, ...]
    maintenance_revision: int | None = Field(default=None, ge=1)
    call_limit: int | None = Field(default=None, ge=1, le=128)


class SpaceView(BaseModel):
    model_config = ConfigDict(from_attributes=True, frozen=True)
    id: UUID
    status: str
    quiesced: bool
    fence: int
    cleanup_attempts: int
    error_code: str | None
    usage: dict | None
    cleaned_at: datetime | None
    execution: dict | None = None
    stop_proof: str | None = None


class WriterIdentity(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    protocol: Literal["mem0-spawn-v1", "graphiti-spawn-v1"] = "mem0-spawn-v1"
    host_id: str = Field(pattern=r"^[a-f0-9]{64}$")
    boot_time: float
    execution_id: UUID
    pid: int | None = Field(default=None, gt=0)
    started_at: float | None = None


class CleanupLease(BaseModel):
    model_config = ConfigDict(frozen=True)
    space_id: UUID
    pair_id: UUID
    backend: str
    fence: int
    worker_id: str


class CleanupProof(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)
    store_empty: bool
    auxiliary_empty: bool


class BuildUsage(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)
    attempted_calls: int = Field(ge=0)
    reported_tokens: int = Field(ge=0)
    usage_complete: bool
    processed_sources: int = Field(ge=0)
    native_changes: int = Field(ge=0)
