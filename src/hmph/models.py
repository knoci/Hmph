from datetime import datetime
from typing import Any
from uuid import UUID, uuid4

from sqlalchemy import (
    BigInteger,
    CheckConstraint,
    DateTime,
    ForeignKey,
    ForeignKeyConstraint,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    func,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class Owner(Base):
    __tablename__ = "owners"
    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    name: Mapped[str] = mapped_column(String(80))


class Character(Base):
    __tablename__ = "characters"
    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    slug: Mapped[str] = mapped_column(String(80), unique=True)
    persona_version: Mapped[int] = mapped_column(default=1)


class Pair(Base):
    __tablename__ = "pairs"
    __table_args__ = (UniqueConstraint("owner_id", "character_id"),)
    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    owner_id: Mapped[UUID] = mapped_column(ForeignKey("owners.id"))
    character_id: Mapped[UUID] = mapped_column(ForeignKey("characters.id"))
    preferences: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict)
    policy_revision: Mapped[int] = mapped_column(default=1)
    memory_epoch: Mapped[int] = mapped_column(default=1)
    next_memory_seq: Mapped[int] = mapped_column(BigInteger, default=1, server_default="1")
    next_framework_seq: Mapped[int] = mapped_column(BigInteger, default=1, server_default="1")
    memory_config_revision: Mapped[int] = mapped_column(default=1, server_default="1")
    selected_memory_build: Mapped[UUID | None] = mapped_column(
        ForeignKey("memory_builds.id", use_alter=True, name="pair_selected_memory_build")
    )


class Chat(Base):
    __tablename__ = "conversations"
    __table_args__ = (UniqueConstraint("id", "pair_id"),)
    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    pair_id: Mapped[UUID] = mapped_column(ForeignKey("pairs.id"), index=True)
    next_source_seq: Mapped[int] = mapped_column(BigInteger, default=1)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class Turn(Base):
    __tablename__ = "turns"
    __table_args__ = (
        ForeignKeyConstraint(
            ["conversation_id", "pair_id"], ["conversations.id", "conversations.pair_id"]
        ),
        CheckConstraint(
            "status IN ('accepted','preparing','generating','completed',"
            "'cancelled','failed','invalidated')",
            name="turn_status",
        ),
        Index(
            "one_active_turn_per_pair",
            "pair_id",
            unique=True,
            postgresql_where=text("status IN ('accepted','preparing','generating')"),
        ),
    )
    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    conversation_id: Mapped[UUID] = mapped_column(index=True)
    pair_id: Mapped[UUID] = mapped_column(ForeignKey("pairs.id"))
    status: Mapped[str] = mapped_column(String(20), default="accepted")
    source_seq: Mapped[int] = mapped_column(BigInteger)
    next_event_id: Mapped[int] = mapped_column(default=1)
    published_segments: Mapped[int] = mapped_column(default=0)
    policy_revision: Mapped[int] = mapped_column()
    memory_epoch: Mapped[int] = mapped_column()
    source_timezone: Mapped[str | None] = mapped_column(String(80))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    deadline: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    error_code: Mapped[str | None] = mapped_column(String(60))
    model_mode: Mapped[str] = mapped_column(String(20), default="fake", server_default="fake")
    usage: Mapped[dict[str, Any] | None] = mapped_column(JSONB)
    content_redacted: Mapped[bool] = mapped_column(default=False, server_default="false")
    memory_context: Mapped[dict[str, Any] | None] = mapped_column(JSONB)
    companion_plan: Mapped[dict[str, Any] | None] = mapped_column(JSONB)


class RepairState(Base):
    __tablename__ = "companion_repairs"
    source_message_id: Mapped[UUID] = mapped_column(ForeignKey("messages.id"), primary_key=True)
    kind: Mapped[str] = mapped_column(String(40), primary_key=True)
    pair_id: Mapped[UUID] = mapped_column(ForeignKey("pairs.id"), index=True)
    source_memory_id: Mapped[UUID] = mapped_column(ForeignKey("memory_entries.id"))
    source_revision: Mapped[int] = mapped_column()
    policy_revision: Mapped[int] = mapped_column()
    status: Mapped[str] = mapped_column(String(20))


class CompanionMention(Base):
    __tablename__ = "companion_mentions"
    turn_id: Mapped[UUID] = mapped_column(ForeignKey("turns.id"), primary_key=True)
    source_message_id: Mapped[UUID] = mapped_column(ForeignKey("messages.id"), primary_key=True)
    source_revision: Mapped[int] = mapped_column()
    segment_index: Mapped[int] = mapped_column()


class Message(Base):
    __tablename__ = "messages"
    __table_args__ = (
        UniqueConstraint("turn_id", "role"),
        UniqueConstraint("conversation_id", "source_seq"),
        CheckConstraint("role IN ('user','assistant')", name="message_role"),
    )
    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    conversation_id: Mapped[UUID] = mapped_column(ForeignKey("conversations.id"))
    turn_id: Mapped[UUID] = mapped_column(ForeignKey("turns.id"))
    source_seq: Mapped[int] = mapped_column(BigInteger)
    role: Mapped[str] = mapped_column(String(20))
    content: Mapped[str] = mapped_column(Text)
    revision: Mapped[int] = mapped_column(default=1)
    deleted: Mapped[bool] = mapped_column(default=False, server_default="false")


class MemoryEntry(Base):
    __tablename__ = "memory_entries"
    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    pair_id: Mapped[UUID] = mapped_column(ForeignKey("pairs.id"), index=True)
    kind: Mapped[str] = mapped_column(String(20))
    predicate: Mapped[str] = mapped_column(String(60))
    scope: Mapped[str] = mapped_column(String(20))
    revision: Mapped[int] = mapped_column(default=1)
    source_order: Mapped[int] = mapped_column(BigInteger)
    status: Mapped[str] = mapped_column(String(20), default="active", server_default="active")


class MemoryVersion(Base):
    __tablename__ = "memory_versions"
    memory_id: Mapped[UUID] = mapped_column(ForeignKey("memory_entries.id"), primary_key=True)
    revision: Mapped[int] = mapped_column(primary_key=True)
    value: Mapped[str] = mapped_column(Text)
    time_text: Mapped[str | None] = mapped_column(String(100))
    reference_start: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    reference_end: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    time_precision: Mapped[str] = mapped_column(
        String(20), default="unknown", server_default="unknown"
    )
    time_basis: Mapped[str] = mapped_column(
        String(30), default="unresolved", server_default="unresolved"
    )
    source_timezone: Mapped[str | None] = mapped_column(String(80))
    recorded_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )


class MemorySource(Base):
    __tablename__ = "memory_sources"
    __table_args__ = (
        ForeignKeyConstraint(
            ["memory_id", "revision"], ["memory_versions.memory_id", "memory_versions.revision"]
        ),
    )
    memory_id: Mapped[UUID] = mapped_column(primary_key=True)
    revision: Mapped[int] = mapped_column(primary_key=True)
    message_id: Mapped[UUID] = mapped_column(ForeignKey("messages.id"), primary_key=True)
    message_revision: Mapped[int] = mapped_column()
    quote: Mapped[str] = mapped_column(Text)


class MemoryTask(Base):
    __tablename__ = "memory_tasks"
    __table_args__ = (
        UniqueConstraint("source_message_id", "source_revision"),
        Index(
            "one_memory_lease_per_pair",
            "pair_id",
            unique=True,
            postgresql_where=text("status = 'leased'"),
        ),
        CheckConstraint(
            "status IN ('pending','leased','done','failed')", name="memory_task_status"
        ),
    )
    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    pair_id: Mapped[UUID] = mapped_column(ForeignKey("pairs.id"), index=True)
    source_message_id: Mapped[UUID] = mapped_column(ForeignKey("messages.id"))
    source_revision: Mapped[int] = mapped_column()
    source_order: Mapped[int] = mapped_column(BigInteger)
    memory_epoch: Mapped[int] = mapped_column()
    model_mode: Mapped[str] = mapped_column(String(20))
    status: Mapped[str] = mapped_column(String(20), default="pending", server_default="pending")
    fence: Mapped[int] = mapped_column(default=0, server_default="0")
    attempts: Mapped[int] = mapped_column(default=0, server_default="0")
    worker_id: Mapped[str | None] = mapped_column(String(100))
    lease_until: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    error_code: Mapped[str | None] = mapped_column(String(60))
    usage: Mapped[dict[str, Any] | None] = mapped_column(JSONB)


class StreamEvent(Base):
    __tablename__ = "stream_events"
    __table_args__ = (UniqueConstraint("turn_id", "segment_index"),)
    turn_id: Mapped[UUID] = mapped_column(ForeignKey("turns.id"), primary_key=True)
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    type: Mapped[str] = mapped_column(String(40))
    payload: Mapped[dict[str, Any]] = mapped_column(JSONB)
    segment_index: Mapped[int | None] = mapped_column()


class Job(Base):
    __tablename__ = "jobs"
    __table_args__ = (CheckConstraint("status IN ('pending','leased','done')", name="job_status"),)
    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    turn_id: Mapped[UUID] = mapped_column(ForeignKey("turns.id"), unique=True)
    status: Mapped[str] = mapped_column(String(20), default="pending", index=True)
    fence: Mapped[int] = mapped_column(default=0)
    worker_id: Mapped[str | None] = mapped_column(String(100))
    lease_until: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class WriteRequest(Base):
    __tablename__ = "write_requests"
    owner_id: Mapped[UUID] = mapped_column(ForeignKey("owners.id"), primary_key=True)
    operation: Mapped[str] = mapped_column(String(40), primary_key=True)
    request_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    digest: Mapped[str] = mapped_column(String(64))
    result_id: Mapped[UUID] = mapped_column()


class TurnSourceUse(Base):
    __tablename__ = "turn_source_uses"
    turn_id: Mapped[UUID] = mapped_column(ForeignKey("turns.id"), primary_key=True)
    message_id: Mapped[UUID] = mapped_column(ForeignKey("messages.id"), primary_key=True)
    message_revision: Mapped[int] = mapped_column()


class TurnMemoryUse(Base):
    __tablename__ = "turn_memory_uses"
    turn_id: Mapped[UUID] = mapped_column(ForeignKey("turns.id"), primary_key=True)
    memory_id: Mapped[UUID] = mapped_column(ForeignKey("memory_entries.id"), primary_key=True)
    revision: Mapped[int] = mapped_column()


class MemoryDeletion(Base):
    __tablename__ = "memory_deletions"
    __table_args__ = (
        CheckConstraint(
            "(memory_id IS NOT NULL) <> (source_message_id IS NOT NULL)",
            name="deletion_one_target",
        ),
    )
    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    pair_id: Mapped[UUID] = mapped_column(ForeignKey("pairs.id"))
    memory_id: Mapped[UUID | None] = mapped_column(ForeignKey("memory_entries.id"))
    source_message_id: Mapped[UUID | None] = mapped_column(ForeignKey("messages.id"))
    status: Mapped[str] = mapped_column(String(20), default="completed")
    memories: Mapped[int] = mapped_column()
    messages: Mapped[int] = mapped_column()
    turns: Mapped[int] = mapped_column()
    dialogue_states: Mapped[int] = mapped_column(default=0, server_default="0")
    framework_spaces: Mapped[int] = mapped_column(default=0, server_default="0")
    completed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True).evaluates_none(), server_default=func.now()
    )


class DialogueState(Base):
    __tablename__ = "dialogue_states"
    conversation_id: Mapped[UUID] = mapped_column(ForeignKey("conversations.id"), primary_key=True)
    pair_id: Mapped[UUID] = mapped_column(ForeignKey("pairs.id"), index=True)
    revision: Mapped[int] = mapped_column(default=0)
    memory_epoch: Mapped[int] = mapped_column()
    coverage_from_seq: Mapped[int] = mapped_column(BigInteger, default=1)
    processed_through_seq: Mapped[int] = mapped_column(BigInteger, default=0)
    data: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict)


class DialogueSource(Base):
    __tablename__ = "dialogue_sources"
    conversation_id: Mapped[UUID] = mapped_column(
        ForeignKey("dialogue_states.conversation_id"), primary_key=True
    )
    message_id: Mapped[UUID] = mapped_column(ForeignKey("messages.id"), primary_key=True)
    message_revision: Mapped[int] = mapped_column()


class DialogueTask(Base):
    __tablename__ = "dialogue_tasks"
    conversation_id: Mapped[UUID] = mapped_column(
        ForeignKey("dialogue_states.conversation_id"), primary_key=True
    )
    requested_through_seq: Mapped[int] = mapped_column(BigInteger)
    model_mode: Mapped[str] = mapped_column(String(20))
    status: Mapped[str] = mapped_column(String(20), default="pending")
    fence: Mapped[int] = mapped_column(default=0)
    attempts: Mapped[int] = mapped_column(default=0)
    worker_id: Mapped[str | None] = mapped_column(String(100))
    lease_until: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    error_code: Mapped[str | None] = mapped_column(String(60))
    usage: Mapped[dict[str, Any] | None] = mapped_column(JSONB)
    __table_args__ = (
        CheckConstraint(
            "status IN ('pending','leased','idle','failed')", name="dialogue_task_status"
        ),
    )


class MemoryBuild(Base):
    __tablename__ = "memory_builds"
    __table_args__ = (
        UniqueConstraint("pair_id", "generation"),
        CheckConstraint("backend IN ('mem0','graphiti')", name="memory_build_backend"),
        CheckConstraint(
            "status IN ('queued','building','ready','retired','failed')", name="memory_build_status"
        ),
        Index(
            "one_ready_memory_build",
            "pair_id",
            "backend",
            unique=True,
            postgresql_where=text("status = 'ready'"),
        ),
    )
    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    pair_id: Mapped[UUID] = mapped_column(ForeignKey("pairs.id"), index=True)
    generation: Mapped[int] = mapped_column(BigInteger)
    memory_epoch: Mapped[int] = mapped_column()
    backend: Mapped[str] = mapped_column(String(20))
    backend_revision: Mapped[str] = mapped_column(String(40))
    model: Mapped[str] = mapped_column(String(40))
    status: Mapped[str] = mapped_column(String(20), default="queued")
    watermark: Mapped[int] = mapped_column(BigInteger)
    source_count: Mapped[int] = mapped_column()
    snapshot_digest: Mapped[str] = mapped_column(String(64))
    fence: Mapped[int] = mapped_column(default=0)
    attempts: Mapped[int] = mapped_column(default=0)
    worker_id: Mapped[str | None] = mapped_column(String(100))
    lease_until: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    error_code: Mapped[str | None] = mapped_column(String(60))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    ready_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class MemoryBuildSource(Base):
    __tablename__ = "memory_build_sources"
    build_id: Mapped[UUID] = mapped_column(ForeignKey("memory_builds.id"), primary_key=True)
    message_id: Mapped[UUID] = mapped_column(ForeignKey("messages.id"), primary_key=True)
    message_revision: Mapped[int] = mapped_column()
    source_order: Mapped[int] = mapped_column(BigInteger)


class MemorySpace(Base):
    __tablename__ = "memory_spaces"
    __table_args__ = (
        UniqueConstraint("build_id", "fence"),
        CheckConstraint(
            "status IN ('writing','ready','retired','cleaning','cleaned')",
            name="memory_space_status",
        ),
    )
    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    build_id: Mapped[UUID] = mapped_column(ForeignKey("memory_builds.id"), index=True)
    status: Mapped[str] = mapped_column(String(20), default="writing")
    fence: Mapped[int] = mapped_column()
    worker_id: Mapped[str] = mapped_column(String(100))
    quiesced: Mapped[bool] = mapped_column(default=False)
    execution: Mapped[dict[str, Any] | None] = mapped_column(JSONB)
    stop_proof: Mapped[str | None] = mapped_column(String(60))
    cleanup_fence: Mapped[int] = mapped_column(default=0)
    cleanup_attempts: Mapped[int] = mapped_column(default=0)
    cleanup_worker_id: Mapped[str | None] = mapped_column(String(100))
    cleanup_until: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    error_code: Mapped[str | None] = mapped_column(String(60))
    usage: Mapped[dict[str, Any] | None] = mapped_column(JSONB)
    cleaned_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class MemoryDeletionSpace(Base):
    __tablename__ = "memory_deletion_spaces"
    deletion_id: Mapped[UUID] = mapped_column(ForeignKey("memory_deletions.id"), primary_key=True)
    space_id: Mapped[UUID] = mapped_column(ForeignKey("memory_spaces.id"), primary_key=True)


class MemoryNativeSource(Base):
    """Native episode IDs map to authoritative source revisions, never text copies."""

    __tablename__ = "memory_native_sources"
    space_id: Mapped[UUID] = mapped_column(ForeignKey("memory_spaces.id"), primary_key=True)
    native_id: Mapped[str] = mapped_column(String(100), primary_key=True)
    message_id: Mapped[UUID] = mapped_column(ForeignKey("messages.id"), index=True)
    message_revision: Mapped[int] = mapped_column()
    source_order: Mapped[int] = mapped_column(BigInteger)


class MemoryRead(Base):
    """Bounded read/publish capability. No source text or framework facts."""

    __tablename__ = "memory_reads"
    turn_id: Mapped[UUID] = mapped_column(ForeignKey("turns.id"), primary_key=True)
    build_id: Mapped[UUID] = mapped_column(ForeignKey("memory_builds.id"), index=True)
    space_id: Mapped[UUID] = mapped_column(ForeignKey("memory_spaces.id"))
    config_revision: Mapped[int] = mapped_column()
    cutoff: Mapped[int] = mapped_column(BigInteger)
    job_fence: Mapped[int] = mapped_column()
    valid_until: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    invalidated: Mapped[bool] = mapped_column(default=False, server_default="false")
