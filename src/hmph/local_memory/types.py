"""Typed records and schemas for the standalone SQLite local memory backend."""

from __future__ import annotations

from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

ItemKind = Literal["plan", "fact", "preference", "episode"]
ItemStatus = Literal["active", "paused", "completed", "cancelled", "retracted"]
EpistemicType = Literal["confirmed_fact", "shared_commitment", "inferred_belief"]
ChangeType = Literal[
    "create",
    "reschedule",
    "update",
    "pause",
    "resume",
    "complete",
    "cancel",
    "retract",
    "ambiguous",
]
EdgeRelation = Literal[
    "SUPERSEDES",
    "RETRACTS",
    "SUPPORTS",
    "CONFLICTS",
    "RELATED_ENTITY",
    "DERIVED_FROM",
]


class MessageRecord(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: str
    session_id: str
    namespace: str = "default"
    role: Literal["user", "assistant"]
    content: str
    created_at: str
    timezone: str
    extracted: bool = False
    deleted: bool = False


class VersionRecord(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: str
    item_id: str
    version_no: int
    summary: str
    status: ItemStatus
    effective_time_text: str | None = None
    valid_from: str | None = None
    valid_to: str | None = None
    condition_text: str | None = None
    change_type: str
    supersedes_version_id: str | None = None
    source_message_id: str
    source_quote: str
    source_content: str | None = None
    created_at: str


class ItemRecord(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: str
    namespace: str = "default"
    title: str
    kind: ItemKind
    status: ItemStatus
    current_version_id: str | None = None
    entities: tuple[str, ...] = ()
    epistemic_type: EpistemicType = "confirmed_fact"
    confidence: float = 1.0
    salience: float = 1.0
    last_mentioned_at: str | None = None
    created_at: str
    updated_at: str
    versions: tuple[VersionRecord, ...] = ()

    @property
    def current(self) -> VersionRecord | None:
        if not self.versions:
            return None
        if self.current_version_id:
            for version in self.versions:
                if version.id == self.current_version_id:
                    return version
        return self.versions[-1]


class EdgeRecord(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: str
    src_type: str
    src_id: str
    dst_type: str
    dst_id: str
    relation: EdgeRelation
    source_message_id: str | None = None
    created_at: str


class AmbiguityRecord(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: str
    namespace: str = "default"
    source_message_id: str
    source_quote: str
    candidate_item_ids: tuple[str, ...] = ()
    note: str
    resolved: bool = False
    created_at: str


class ExtractedChange(BaseModel):
    model_config = ConfigDict(extra="ignore")

    action: ChangeType
    target_item_id: str | None = None
    title: str | None = None
    kind: ItemKind = "plan"
    summary: str = ""
    effective_time_text: str | None = None
    valid_from: str | None = None
    valid_to: str | None = None
    condition_text: str | None = None
    entities: list[str] = Field(default_factory=list)
    related_item_ids: list[str] = Field(default_factory=list)
    candidate_item_ids: list[str] = Field(default_factory=list)
    ambiguity_note: str | None = None
    source_quote: str = ""
    source_turn_index: int | None = None
    epistemic_type: EpistemicType = "confirmed_fact"
    confidence: float = 1.0


class ExtractionResponse(BaseModel):
    model_config = ConfigDict(extra="ignore")

    changes: list[ExtractedChange] = Field(default_factory=list)


class TurnEvidence(BaseModel):
    model_config = ConfigDict(frozen=True)

    namespace: str = "default"
    query_intent: Literal["current", "history", "both", "casual"]
    embedding_backend: str
    used_item_ids: tuple[str, ...] = ()
    used_source_message_ids: tuple[str, ...] = ()
    used_consolidation_ids: tuple[str, ...] = ()
    used_session_summary_ids: tuple[str, ...] = ()
    pending_message_ids: tuple[str, ...] = ()
    ambiguity_ids: tuple[str, ...] = ()
    narrated_context: str = ""
    retrieval_ms: float = 0.0


class TurnResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    session_id: str
    user_message_id: str
    assistant_message_id: str
    reply: str
    evidence: TurnEvidence
    generation_ms: float = 0.0
    usage: dict[str, int] | None = None


class ForgetReceipt(BaseModel):
    model_config = ConfigDict(frozen=True)

    query: str
    namespace: str = "default"
    deleted_item_ids: tuple[str, ...] = ()
    deleted_version_ids: tuple[str, ...] = ()
    deleted_message_ids: tuple[str, ...] = ()
    deleted_ambiguity_ids: tuple[str, ...] = ()
    deleted_consolidation_ids: tuple[str, ...] = ()
    deleted_session_summary_ids: tuple[str, ...] = ()
    cleared_core_blocks: tuple[str, ...] = ()
    deleted_edge_count: int = 0


class SessionSummaryRecord(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: str
    namespace: str = "default"
    session_id: str
    session_date: str
    summary: str
    facts: tuple[str, ...] = ()
    entities: tuple[str, ...] = ()
    source_message_ids: tuple[str, ...] = ()
    updated_at: str


class CoreMemoryBlock(BaseModel):
    model_config = ConfigDict(frozen=True)

    namespace: str = "default"
    block_key: str
    content: str
    source_item_ids: tuple[str, ...] = ()
    updated_at: str


class ProactiveSuggestion(BaseModel):
    model_config = ConfigDict(frozen=True)

    namespace: str = "default"
    should_initiate: bool = False
    reason: Literal["none", "open_ambiguity", "active_plan_checkin"] = "none"
    item_id: str | None = None
    topic: str | None = None
    suggested_cue: str | None = None
    excluded_paused_count: int = 0


class MemoryEvent(BaseModel):
    model_config = ConfigDict(frozen=True)

    event_type: Literal[
        "memory_extracted",
        "item_updated",
        "ambiguity_detected",
        "consolidation_completed",
        "memory_forgotten",
        "core_memory_updated",
        "session_summarized",
    ]
    namespace: str = "default"
    timestamp: str
    payload: dict = Field(default_factory=dict)


class ConsolidationRecord(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: str
    namespace: str = "default"
    topic: str
    summary: str
    source_item_ids: tuple[str, ...] = ()
    source_message_ids: tuple[str, ...] = ()
    source_quotes: tuple[str, ...] = ()
    confidence: float = 0.85
    updated_at: str


class ConsolidationReport(BaseModel):
    model_config = ConfigDict(frozen=True)

    namespace: str = "default"
    merged_duplicates: int = 0
    decayed_items: int = 0
    consolidations: tuple[ConsolidationRecord, ...] = ()
    usage: dict[str, int] | None = None


def is_uuid(value: str | None) -> bool:
    if not value:
        return False
    try:
        UUID(value)
        return True
    except ValueError:
        return False
