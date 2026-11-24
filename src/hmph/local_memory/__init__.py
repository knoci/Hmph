"""Standalone single-process SQLite local memory backend for Hmph."""

from hmph.local_memory.embeddings import (
    Embedder,
    HashingEmbedder,
    LocalFastEmbedder,
    OpenAIEmbedder,
    build_default_embedder,
)
from hmph.local_memory.engine import LocalMemoryEngine
from hmph.local_memory.gateway import create_airi_gateway_app
from hmph.local_memory.llm import LLMClient, OpenAICompatLLM, resolve_llm_config
from hmph.local_memory.mcp_server import LocalMCPServer
from hmph.local_memory.store import LocalMemoryStore
from hmph.local_memory.types import (
    AmbiguityRecord,
    ConsolidationRecord,
    ConsolidationReport,
    CoreMemoryBlock,
    EdgeRecord,
    ExtractedChange,
    ExtractionResponse,
    ForgetReceipt,
    ItemRecord,
    MemoryEvent,
    MessageRecord,
    ProactiveSuggestion,
    SessionSummaryRecord,
    TurnEvidence,
    TurnResult,
    VersionRecord,
)

__all__ = [
    "AmbiguityRecord",
    "ConsolidationRecord",
    "ConsolidationReport",
    "CoreMemoryBlock",
    "EdgeRecord",
    "Embedder",
    "ExtractedChange",
    "ExtractionResponse",
    "ForgetReceipt",
    "HashingEmbedder",
    "ItemRecord",
    "LLMClient",
    "LocalFastEmbedder",
    "LocalMCPServer",
    "LocalMemoryEngine",
    "LocalMemoryStore",
    "MemoryEvent",
    "MessageRecord",
    "OpenAICompatLLM",
    "OpenAIEmbedder",
    "ProactiveSuggestion",
    "SessionSummaryRecord",
    "TurnEvidence",
    "TurnResult",
    "VersionRecord",
    "build_default_embedder",
    "create_airi_gateway_app",
    "resolve_llm_config",
]
