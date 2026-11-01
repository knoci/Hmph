"""Evaluation input boundary. Gold data never belongs to ModelInput."""

import hashlib
import json
import subprocess
from collections import Counter
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict


class HistoryMessage(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    source_id: str
    session_id: str
    timestamp: str | None
    role: Literal["user", "assistant"]
    content: str


class QueryInput(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    question: str
    timestamp: str | None


class ModelInput(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    history: tuple[HistoryMessage, ...]
    query: QueryInput


@dataclass(frozen=True)
class GoldAnswer:
    answer: Any
    evidence_sessions: tuple[str, ...]
    category: str | None
    abstention: bool
    evidence_messages: tuple[str, ...] = ()


@dataclass(frozen=True)
class EvaluationCase:
    case_id: str
    history: tuple[HistoryMessage, ...]
    query: QueryInput
    gold: GoldAnswer

    def model_input(self) -> ModelInput:
        return ModelInput(history=self.history, query=self.query)


def parse_longmemeval(raw: dict) -> EvaluationCase:
    """Whitelist the original history fields; even nested evidence labels are discarded.

    Dates are preserved in the dataset's original format at E0. Temporal
    normalization and actual answering are deliberately not performed here.
    """
    sessions = raw["haystack_sessions"]
    dates = raw["haystack_dates"]
    original_ids = raw["haystack_session_ids"]
    if not isinstance(sessions, list) or not sessions:
        raise ValueError("Expected non-empty history sessions")
    if not (len(sessions) == len(dates) == len(original_ids)):
        raise ValueError("History sessions, IDs and dates must align")
    # Cleaned-S includes repeated filler IDs at different dates. Preserve every
    # occurrence with distinct neutral IDs. Ambiguous gold references still fail.
    counts = Counter(original_ids)
    if any(counts.get(item, 0) > 1 for item in raw.get("answer_session_ids", [])):
        raise ValueError("Ambiguous duplicate evidence session ID")
    session_mapping = {original: f"s{index:05d}" for index, original in enumerate(original_ids)}
    history = []
    evidence_messages = []
    for index, (messages, date) in enumerate(zip(sessions, dates, strict=True)):
        if not isinstance(messages, list):
            raise ValueError("Expected a list of messages")
        for message_index, message in enumerate(messages):
            if message.get("has_answer") is True:
                evidence_messages.append(f"s{index:05d}/m{message_index:05d}")
            history.append(
                HistoryMessage(
                    source_id=f"s{index:05d}/m{message_index:05d}",
                    session_id=f"s{index:05d}",
                    timestamp=date,
                    role=message["role"],
                    content=message["content"],
                )
            )
    evidence = raw.get("answer_session_ids", [])
    if any(session_id not in session_mapping for session_id in evidence):
        raise ValueError("Gold evidence refers to an unknown session")
    case_id = str(raw["question_id"])
    return EvaluationCase(
        case_id=case_id,
        history=tuple(history),
        query=QueryInput(question=raw["question"], timestamp=raw.get("question_date")),
        gold=GoldAnswer(
            answer=raw["answer"],
            evidence_sessions=tuple(session_mapping[item] for item in evidence),
            category=raw.get("question_type"),
            abstention=case_id.endswith("_abs"),
            evidence_messages=tuple(evidence_messages),
        ),
    )


def _code_version() -> dict:
    root = Path(__file__).resolve().parents[2]
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=root,
                capture_output=True,
                text=True,
                timeout=5,
                check=True,
            ).stdout.strip()
        )
        return {"commit": commit, "dirty": dirty}
    except (OSError, subprocess.SubprocessError):
        return {"commit": None, "dirty": None}


def audit_suite(path: Path) -> dict:
    if path.stat().st_size > 32 * 1024 * 1024:
        raise ValueError("E0 audits support suites up to 32 MiB; streaming datasets are not ready")
    raw_bytes = path.read_bytes()
    data = json.loads(raw_bytes)
    if not isinstance(data, list) or not data:
        raise ValueError("Suite must be a non-empty list")
    cases = [parse_longmemeval(item) for item in data]
    if len({case.case_id for case in cases}) != len(cases):
        raise ValueError("Duplicate evaluation case IDs")
    messages = 0
    for case in cases:
        exposed = case.model_input()
        ModelInput.model_validate_json(exposed.model_dump_json())
        messages += len(exposed.history)
    return {
        "schema_version": "hmph-eval-v1",
        "mode": "input-audit",
        "created_at": datetime.now(UTC).isoformat(),
        "code": _code_version(),
        "dataset_sha256": hashlib.sha256(raw_bytes).hexdigest(),
        "adapter": "longmemeval-history-whitelist-v2",
        "case_count": len(cases),
        "source_message_count": messages,
        "input_audit": {
            "status": "passed",
            "checks": [
                "schema-validation",
                "aligned-sessions",
                "opaque-model-visible-ids",
                "history-field-whitelist",
                "separate-gold-structure",
            ],
        },
        "quality_status": "not_run",
        "quality_metrics": None,
        "inference_requests": 0,
        "limitations": [
            "No model inference or memory benchmark was run.",
            "Dates preserved; temporal normalization not evaluated.",
            "Public benchmark pretraining contamination cannot be ruled out.",
        ],
    }
