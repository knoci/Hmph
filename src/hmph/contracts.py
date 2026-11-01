from datetime import datetime
from typing import Any
from uuid import UUID

from pydantic import BaseModel, ConfigDict

ACTIVE = ("accepted", "preparing", "generating")
TERMINAL = ("completed", "cancelled", "failed", "invalidated")


class DomainError(Exception):
    def __init__(self, code: str, status: int = 409):
        self.code = code
        self.status = status
        super().__init__(code)


class Identity(BaseModel):
    owner_id: UUID
    character_id: UUID


class View(BaseModel):
    model_config = ConfigDict(from_attributes=True, frozen=True)


class ChatView(View):
    id: UUID
    created_at: datetime


class TurnView(View):
    id: UUID
    conversation_id: UUID
    status: str
    next_event_id: int
    error_code: str | None
    model_mode: str
    usage: dict[str, Any] | None
    content_redacted: bool


class MessageView(View):
    id: UUID
    turn_id: UUID
    role: str
    content: str
    source_seq: int
    deleted: bool
    revision: int


class EventView(View):
    turn_id: UUID
    id: int
    type: str
    payload: dict[str, Any]


class Lease(View):
    turn_id: UUID
    owner_id: UUID
    worker_id: str
    fence: int
    policy_revision: int
    memory_epoch: int
    deadline: datetime
    remaining_seconds: float
    user_content: str
