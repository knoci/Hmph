"""One literal retrieval bridge for opt-in, bounded product experiments.

This chooses navigation, never facts. The caller still owns the authoritative
source, policy, deletion, lease, candidate and context-budget checks.
"""

import re

from pydantic import BaseModel, ConfigDict, Field

from hmph.model_gateway import GenerationRequest, ModelUsage
from hmph.working_memory import FocusPlan


class RecallSource(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    source_id: str
    revision: int
    content: str


class RecallInput(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    question: str = Field(min_length=1, max_length=4000)
    missing_slots: tuple[str, ...]
    sources: tuple[RecallSource, ...] = Field(max_length=12)


class RecallDecision(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    plan: FocusPlan
    usage: ModelUsage | None = None


class RecallError(Exception):
    def __init__(self, status, usage=None):
        if status not in {"invalid", "input_limit"}:
            raise ValueError("Invalid recall failure status")
        self.status, self.usage = status, usage
        super().__init__(status)


def prepare_recall(request: GenerationRequest):
    workspace, companion = request.workspace, request.companion
    query = request.messages[-1].content
    if (
        workspace is None
        or workspace.plan.intent == "social"
        or workspace.coverage == "complete"
        or workspace.delta_gap
        or workspace.policy_filtered_sources
        or request.short_term.pending_gap
        or request.short_term.state_truncated
        or companion is None
        or companion.act in {"close", "repair", "clarify"}
        or companion.repair_action != "none"
        or companion.shared_thread_action == "pause"
        or companion.thread_control is not None
        or re.search(r"改成|改到|不是|取消|更正|纠正|instead|correction", query, re.I)
    ):
        return None
    permissions = {
        p.source_id for p in companion.memory_use if p.permission in {"context_only", "mention"}
    }
    sources = {}
    for raw in request.raw_evidence:
        if raw.role == "user" and raw.message_id in permissions:
            sources[str(raw.message_id)] = RecallSource(
                source_id=str(raw.message_id), revision=raw.message_revision, content=raw.content
            )
    for memory in request.evidence:
        for source in memory.sources:
            if source.message_id in permissions:
                sources.setdefault(
                    str(source.message_id),
                    RecallSource(
                        source_id=str(source.message_id),
                        revision=source.message_revision,
                        content=source.quote,
                    ),
                )
    missing = tuple(s.name for s in workspace.slots if s.status != "supported")
    if not sources or not missing:
        return None
    return RecallInput(
        question=query,
        missing_slots=missing,
        sources=tuple(sources.values())[:12],
    )
