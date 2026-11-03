"""The fake adapter proves transport behavior, not model or memory quality."""

import asyncio
from collections.abc import AsyncGenerator
from datetime import datetime
from typing import Literal, Protocol
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from hmph.companion_policy import CompanionPlan
from hmph.dialogue_types import ShortTermContext
from hmph.memory_types import MemoryView
from hmph.preference_types import ExpressionPreferences
from hmph.working_memory import EvidenceWorkspace, RawEvidence

FAKE_SENTENCES = ("哼，收到啦。", "我现在使用的是测试模型，还没有接入真实记忆。")
FALLBACK = "这条回复暂时无法提供，请稍后再试。"


class ChatMessage(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    role: Literal["user", "assistant"]
    content: str


class GenerationRequest(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    messages: tuple[ChatMessage, ...] = Field(min_length=1)
    evidence: tuple[MemoryView, ...] = ()
    short_term: ShortTermContext = Field(default_factory=ShortTermContext)
    raw_evidence: tuple[RawEvidence, ...] = ()
    workspace: EvidenceWorkspace | None = None
    preferences: ExpressionPreferences = Field(default_factory=ExpressionPreferences)
    companion: CompanionPlan | None = None
    reference_time: datetime | None = None
    source_timezone: str | None = None
    history_origins: tuple["ContextOrigin", ...] = ()


class ContextOrigin(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    message_id: UUID
    message_revision: int
    start_offset: int | None
    end_offset: int | None


class ModelUsage(BaseModel):
    model_config = ConfigDict(frozen=True, extra="ignore")
    prompt_tokens: int = Field(ge=0)
    completion_tokens: int = Field(ge=0)
    total_tokens: int = Field(ge=0)


class ModelDelta(BaseModel):
    model_config = ConfigDict(frozen=True)
    text: str = ""
    usage: ModelUsage | None = None


class GuardReview(BaseModel):
    model_config = ConfigDict(frozen=True)
    allowed: bool
    reason: Literal["ok", "attack", "dependency", "distress", "fabricated_action", "other"]
    usage: ModelUsage | None = None
    mentions: tuple["SourceMention", ...] = Field(default=(), max_length=2)


class SourceMention(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    source_id: UUID
    quote: str = Field(min_length=2, max_length=80)
    # The reviewer links two independently literal spans. A translated or
    # normalized-date reply need not copy the source's surface form.
    source_quote: str | None = Field(default=None, min_length=2, max_length=100)


class ModelGateway(Protocol):
    def stream(self, request: GenerationRequest) -> AsyncGenerator[ModelDelta, None]: ...


class FakeModel:
    def __init__(self, *, delay: float = 0.01):
        self.delay = delay

    async def stream(self, request: GenerationRequest) -> AsyncGenerator[ModelDelta, None]:
        for chunk in ("哼，收", "到啦。", "我现在使用的是测试模型，", "还没有接入真实记忆。"):
            await asyncio.sleep(self.delay)
            yield ModelDelta(text=chunk)


class FakeOutputGuard:
    """Allow only pre-reviewed fixture sentences. Never use for a real model."""

    async def allows(self, prefix: str, candidate: str) -> bool:
        return candidate in (*FAKE_SENTENCES, FALLBACK)
