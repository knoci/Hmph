"""P2 trajectory protocol. Gold never crosses the writer/reader interfaces."""

import hashlib
import json
import re
from collections import Counter
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Literal
from zoneinfo import ZoneInfo

from pydantic import BaseModel, ConfigDict, Field, model_validator

from hmph.selection_eval import CATEGORIES, Gold, Query, SelectionCase, Source, grade


class RunConfiguration(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    protocol: Literal["p1", "p2a"]
    profiles: tuple[Literal["raw", "mem0", "graphiti"], ...]
    case_ids: tuple[str, ...] = ()
    max_calls: int = Field(gt=0)
    max_input_bytes: int = Field(gt=0)
    per_case_calls: int = Field(gt=0)
    reader_bytes: int = Field(ge=1024, le=12000)
    evidence_mode: Literal["native", "source_only"]

    @property
    def limits(self):
        return {
            key: getattr(self, key) for key in ("max_calls", "max_input_bytes", "per_case_calls")
        }

    @classmethod
    def for_protocol(cls, protocol, **overrides):
        defaults = (
            dict(
                profiles=("raw", "graphiti"),
                max_calls=128,
                max_input_bytes=4 * 1024 * 1024,
                per_case_calls=80,
                reader_bytes=5000,
                evidence_mode="source_only",
            )
            if protocol == "p2a"
            else dict(
                profiles=("raw", "mem0", "graphiti"),
                max_calls=384,
                max_input_bytes=8 * 1024 * 1024,
                per_case_calls=24,
                reader_bytes=12000,
                evidence_mode="native",
            )
        )
        return cls(protocol=protocol, **{**defaults, **overrides})

    @model_validator(mode="after")
    def bounded_protocol(self):
        if self.protocol == "p2a":
            if (
                self.profiles != ("raw", "graphiti")
                or len(self.case_ids) != 2
                or len(set(self.case_ids)) != 2
                or any(re.fullmatch(r"p2-dev-[0-9]{3}", c) is None for c in self.case_ids)
                or self.evidence_mode != "source_only"
                or self.reader_bytes > 5000
                or self.max_calls > 128
                or self.max_input_bytes > 4 * 1024 * 1024
                or self.per_case_calls > 80
            ):
                raise ValueError(
                    "P2-a requires two explicit dev cases and bounded source-only profiles"
                )
        elif (
            self.profiles != ("raw", "mem0", "graphiti")
            or self.case_ids
            or self.max_calls > 384
            or self.max_input_bytes > 8 * 1024 * 1024
            or self.per_case_calls > 24
        ):
            raise ValueError("P1 configuration exceeds its frozen protocol")
        return self


class SourceMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    session_id: str = Field(min_length=1, max_length=100)
    source_order: int = Field(gt=0)
    source_timezone: str = Field(min_length=1, max_length=80)


class TrajectoryRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    id: str = Field(pattern=r"^p2-(dev|holdout)-[0-9]{3}$")
    person_id: str = Field(min_length=1, max_length=100)
    category: str
    history: tuple[Source, ...] = Field(min_length=10, max_length=100)
    query: Query
    gold: Gold
    source_metadata: dict[str, SourceMetadata]
    build_watermark: int = Field(gt=0)
    conditions: tuple[str, ...]
    evidence_alternatives: tuple[tuple[str, ...], ...]


class TrajectorySuite(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    schema_version: Literal["hmph-p2-trajectories-v1"]
    split: Literal["development", "holdout"]
    status: Literal["draft", "frozen"]
    cases: tuple[TrajectoryRecord, ...]


@dataclass(frozen=True)
class TrajectoryCase(SelectionCase):
    source_metadata: dict[str, SourceMetadata]
    build_watermark: int
    evidence_alternatives: tuple[tuple[str, ...], ...]

    def writer_input(self):
        return {
            "history": [
                source.model_dump(mode="json")
                for source in self.history
                if self.source_metadata[source.source_id].source_order <= self.build_watermark
            ]
        }


def load_trajectories(path, *, require_frozen=False):
    raw = Path(path).read_bytes()
    suite = TrajectorySuite.model_validate_json(raw)
    if require_frozen and suite.status != "frozen":
        raise ValueError("Trajectory suite is not frozen")
    quota = 2 if suite.split == "development" else 3
    if Counter(case.category for case in suite.cases) != Counter({c: quota for c in CATEGORIES}):
        raise ValueError("P2 requires two development or three holdout cases per category")
    if len({case.person_id for case in suite.cases}) != len(suite.cases):
        raise ValueError("Trajectory persons must be independent")
    cases = []
    for index, case in enumerate(suite.cases, 1):
        prefix = "dev" if suite.split == "development" else "holdout"
        ids = {s.source_id for s in case.history}
        metadata = case.source_metadata
        orders = [
            metadata[s.source_id].source_order for s in case.history if s.source_id in metadata
        ]
        evidence_sets = (case.gold.evidence_sources, *case.evidence_alternatives)
        if (
            case.id != f"p2-{prefix}-{index:03d}"
            or len(ids) != len(case.history)
            or set(metadata) != ids
            or len(set(orders)) != len(case.history)
            or orders != sorted(orders)
            or not any(order <= case.build_watermark for order in orders)
            or not any(order > case.build_watermark for order in orders)
            or len({m.session_id for m in metadata.values()}) < 2
            or list(case.history) != sorted(case.history, key=lambda s: s.observed_at)
            or any(s.observed_at > case.query.timestamp for s in case.history)
            or any(
                not set(group).issubset(ids) or len(set(group)) != len(group)
                for group in evidence_sets
            )
            or (case.gold.answer is None and any(evidence_sets))
            or (case.gold.answer is not None and any(not group for group in evidence_sets))
        ):
            raise ValueError("Invalid trajectory history, watermark or evidence")
        for meta in metadata.values():
            ZoneInfo(meta.source_timezone)
        cases.append(
            TrajectoryCase(
                case.id,
                case.category,
                case.history,
                case.query,
                case.gold,
                metadata,
                case.build_watermark,
                case.evidence_alternatives,
            )
        )
    return cases, hashlib.sha256(raw).hexdigest()


def load_suite(path, *, require_frozen=False):
    from hmph.selection_eval import load_selection

    schema = json.loads(Path(path).read_bytes()).get("schema_version")
    if schema == "hmph-p2-trajectories-v1":
        return load_trajectories(path, require_frozen=require_frozen)
    return load_selection(path)


def source_context(case, retrieved):
    """Resolve actual source IDs, discard derived assertions, then add unprocessed delta.

    This evaluation-only overlay simulates the known watermark. It is not the
    product's persistent reader/cleanup coordination, which belongs to P2-c.
    """
    sources = {s.source_id: s for s in case.history}
    delta = []
    if isinstance(case, TrajectoryCase):
        delta = [
            s.source_id
            for s in case.history
            if case.source_metadata[s.source_id].source_order > case.build_watermark
        ]
    selected = list(delta)
    for item in retrieved:
        if item["kind"] != "source":
            continue
        ids = item["source_ids"]
        if len(ids) != 1 or ids[0] not in sources:
            raise ValueError("Unknown native source mapping")
        if isinstance(case, TrajectoryCase) and ids[0] in delta:
            raise ValueError("Native retrieval returned a source beyond its build watermark")
        if ids[0] not in selected:
            selected.append(ids[0])
    return [
        {
            "kind": "source",
            "source_ids": [identifier],
            "content": sources[identifier].content,
            "observed_at": sources[identifier].observed_at.isoformat(),
            "provenance": "original_source",
        }
        for identifier in selected
    ]


def grade_trajectory(case, reply, supplied_sources):
    result = grade(case, reply, supplied_sources)
    if not isinstance(case, TrajectoryCase):
        return result
    for alternative in case.evidence_alternatives:
        candidate = grade(
            replace(case, gold=case.gold.model_copy(update={"evidence_sources": alternative})),
            reply,
            supplied_sources,
        )
        if candidate["required_evidence_covered"]:
            return candidate
    return result
