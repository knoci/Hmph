"""Memory contracts: assertions remain user-stated, never verified world facts."""

from datetime import datetime
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class MemoryProposal(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    kind: Literal["fact", "episode"] = "fact"
    predicate: str = Field(pattern=r"^[a-z][a-z0-9_]{0,59}$")
    value: str = Field(min_length=1, max_length=160)
    quote: str = Field(min_length=1, max_length=300)
    scope: Literal["current", "historical"] = "current"
    time_text: str | None = Field(default=None, max_length=100)


class SourceView(BaseModel):
    model_config = ConfigDict(from_attributes=True, frozen=True)
    message_id: UUID
    message_revision: int
    quote: str


class ReferencedTime(BaseModel):
    start: datetime
    end: datetime
    precision: str
    basis: str
    source_timezone: str


class MemoryView(BaseModel):
    id: UUID
    revision: int
    kind: str
    predicate: str
    scope: str
    value: str
    time_text: str | None
    referenced_time: ReferencedTime | None = None
    recorded_at: datetime
    sources: tuple[SourceView, ...]
    total_sources: int
    epistemic_status: Literal["user_stated"] = "user_stated"


class MemoryLease(BaseModel):
    id: UUID
    pair_id: UUID
    fence: int
    worker_id: str
    memory_epoch: int
    content: str


class LearnResult(BaseModel):
    status: Literal["committed", "stale"]
    written: int = 0


class ForgetPreview(BaseModel):
    memory_id: UUID
    revision: int
    memories: int
    messages: int
    turns: int
    dialogue_states: int = 0
    deletion_mode: Literal["whole_message_and_derived_replies"] = (
        "whole_message_and_derived_replies"
    )


class DeletionReceipt(BaseModel):
    model_config = ConfigDict(from_attributes=True, frozen=True)
    id: UUID
    memory_id: UUID
    status: Literal["completed"]
    memories: int
    messages: int
    turns: int
    dialogue_states: int = 0
    completed_at: datetime


class MemoryTaskView(BaseModel):
    model_config = ConfigDict(from_attributes=True, frozen=True)
    id: UUID
    source_message_id: UUID
    status: str
    attempts: int
    error_code: str | None
    usage: dict | None
