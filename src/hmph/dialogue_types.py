"""Extractive state atoms, not free-form summaries or hidden reasoning traces."""

from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class Anchor(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    message_id: UUID
    text: str = Field(min_length=1, max_length=120)
    quote: str = Field(min_length=1, max_length=240)


class GroundedText(Anchor):
    message_revision: int
    source_seq: int


class Transition(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    item_id: UUID
    status: Literal["resolved", "dropped"]
    evidence: Anchor


class ModeCue(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    mode: Literal["listen", "advice", "solve", "celebrate"]
    evidence: Anchor


class ControlCue(BaseModel):
    """A proposal from the existing state updater, never an active preference."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    kind: Literal["preference", "boundary", "thread", "clarify"]
    field: Literal["nickname", "reply_length", "playfulness", "support_mode"] | None = None
    value: str | None = Field(default=None, max_length=40)
    operation: Literal["create", "progress", "pause", "resume", "resolve", "drop"] | None = None
    target: str | None = Field(default=None, min_length=1, max_length=120)
    mode: Literal["never", "on_request"] | None = None
    evidence: Anchor


class StatePatch(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    topic: Anchor | None = None
    goal: Anchor | None = None
    entities: list[Anchor] = Field(default_factory=list, max_length=6)
    open_items: list[Anchor] = Field(default_factory=list, max_length=6)
    transitions: list[Transition] = Field(default_factory=list, max_length=6)
    response_mode: ModeCue | None = None
    controls: list[ControlCue] = Field(default_factory=list, max_length=4)


class OpenItem(BaseModel):
    id: UUID
    content: GroundedText
    status: Literal["open", "resolved", "dropped"] = "open"
    transition_evidence: GroundedText | None = None


class ResponseMode(BaseModel):
    mode: str
    evidence: GroundedText
    expires_after_seq: int


class StateData(BaseModel):
    model_config = ConfigDict(extra="forbid")
    topic: GroundedText | None = None
    goal: GroundedText | None = None
    entities: list[GroundedText] = Field(default_factory=list)
    open_items: list[OpenItem] = Field(default_factory=list)
    response_mode: ResponseMode | None = None


class StateMessage(BaseModel):
    id: UUID
    revision: int
    source_seq: int
    role: str
    content: str


class SharedThreadHint(BaseModel):
    topic: str
    status: Literal["open", "paused"]
    revision: int


class StateLease(BaseModel):
    conversation_id: UUID
    pair_id: UUID
    memory_epoch: int
    revision: int
    through_seq: int
    worker_id: str
    fence: int
    data: StateData
    messages: tuple[StateMessage, ...]
    policy_revision: int = 0
    shared_threads: tuple[SharedThreadHint, ...] = ()


class StateResult(BaseModel):
    status: Literal["committed", "stale"]


class DialogueView(BaseModel):
    conversation_id: UUID
    revision: int
    processed_through_seq: int
    coverage_from_seq: int
    data: StateData
    pending_messages: int
    task_status: str
    error_code: str | None
    usage: dict | None = None


class ShortTermContext(BaseModel):
    revision: int = 0
    processed_through_seq: int = 0
    coverage_from_seq: int = 1
    data: StateData = Field(default_factory=StateData)
    pending_gap: bool = False
    omitted_delta_messages: int = 0
    state_truncated: bool = False
