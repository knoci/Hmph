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
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    deadline: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    error_code: Mapped[str | None] = mapped_column(String(60))


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
