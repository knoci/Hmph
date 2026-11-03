"""E1: streaming inputs, matched reader budgets, isolated grading and failure accounting.

These are deliberately named baselines, not simplified copies of research systems.
"""

import asyncio
import hashlib
import heapq
import json
import math
import re
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

import ijson
from pydantic import BaseModel, ConfigDict, Field

from hmph.evaluation import QueryInput, _code_version, parse_longmemeval
from hmph.lexical import STOP_WORDS as STOP_WORDS
from hmph.lexical import bm25_order as bm25_order
from hmph.lexical import tokenize
from hmph.model_gateway import ModelUsage
from hmph.semantic import reciprocal_rank_fusion
from hmph.source_context import context_window


class EvidenceChunk(BaseModel):
    source_id: str
    session_id: str
    timestamp: str | None
    role: str
    content: str
    start_offset: int
    end_offset: int
    truncated: bool


class ReaderInput(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    query: QueryInput
    sources: tuple[EvidenceChunk, ...]
    coverage: Literal["subset", "full"]


class QAReply(BaseModel):
    model_config = ConfigDict(extra="forbid")
    answer: str = Field(max_length=2000)
    abstained: bool
    source_ids: tuple[str, ...] = ()
    usage: ModelUsage | None = None


class GradeInput(BaseModel):
    question: str
    reference_answer: str
    hypothesis: str
    abstention_expected: bool


class Grade(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    correct: bool
    usage: ModelUsage | None = None


def load_cases(path: Path, limit, seed):
    if not 1 <= limit <= 500:
        raise ValueError("E1 supports 1..500 cases")
    if path.stat().st_size > 512 * 1024 * 1024:
        raise ValueError("E1 input exceeds 512 MiB")
    checksum = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            checksum.update(chunk)
    heap, ids = [], set()
    with path.open("rb") as source:
        for raw in ijson.items(source, "item"):
            case = parse_longmemeval(raw)
            if case.case_id in ids:
                raise ValueError("Duplicate evaluation case ID")
            ids.add(case.case_id)
            # Selection depends only on seed and ID, never answer or category.
            rank = int(hashlib.sha256(f"{seed}:{case.case_id}".encode()).hexdigest(), 16)
            entry = (-rank, case.case_id, case)
            if len(heap) < limit:
                heapq.heappush(heap, entry)
            elif rank < -heap[0][0]:
                heapq.heapreplace(heap, entry)
    if not heap:
        raise ValueError("Expected nonempty JSON array")
    selected = [v[2] for v in sorted(heap, key=lambda v: -v[0])]
    return selected, checksum.hexdigest(), len(ids)


def _timestamp(value):
    if value is None:
        return (0, 0, 0, 0, 0)
    match = re.search(r"(\d{4})[/-](\d\d)[/-](\d\d)(?:.*?(\d\d):(\d\d))?", value)
    if match is None:
        raise ValueError("Unsupported history date format")
    return tuple(int(v or 0) for v in match.groups())


def chunks(history):
    result = []
    for message in sorted(history, key=lambda m: (_timestamp(m.timestamp), m.source_id)):
        if not message.content:
            continue
        for index, start in enumerate(range(0, len(message.content), 320)):
            end = min(len(message.content), start + 400)
            result.append(
                EvidenceChunk(
                    source_id=f"{message.source_id}/c{index:05d}",
                    session_id=message.session_id,
                    timestamp=message.timestamp,
                    role=message.role,
                    content=message.content[start:end],
                    start_offset=start,
                    end_offset=end,
                    truncated=start > 0 or end < len(message.content),
                )
            )
            if end == len(message.content):
                break
    return result


def _tokens(text):
    return tokenize(text)


def prepare_input(model_input, profile, budget_bytes, semantic_ranker=None):
    if not 512 <= budget_bytes <= 24000:
        raise ValueError("E1 reader budget must be 512..24000 UTF-8 bytes")
    documents = chunks(model_input.history)
    payload = ReaderInput(query=model_input.query, sources=(), coverage="subset")
    if profile == "full_history":
        # Nonoverlapping, complete messages for the genuine full-history condition.
        full = tuple(
            EvidenceChunk(
                source_id=m.source_id,
                session_id=m.session_id,
                timestamp=m.timestamp,
                role=m.role,
                content=m.content,
                start_offset=0,
                end_offset=len(m.content),
                truncated=False,
            )
            for m in sorted(
                model_input.history, key=lambda m: (_timestamp(m.timestamp), m.source_id)
            )
        )
        payload = payload.model_copy(update={"sources": full, "coverage": "full"})
        return payload if len(payload.model_dump_json().encode()) <= budget_bytes else None
    if profile == "recent_window":
        order = list(reversed(range(len(documents))))
    elif profile in ("lexical_raw", "source_aware_raw"):
        order = bm25_order(documents, model_input.query.question)
    elif profile == "hybrid_raw":
        if semantic_ranker is None:
            raise ValueError("Hybrid profile requires a local semantic ranker")
        order = reciprocal_rank_fusion(
            (
                bm25_order(documents, model_input.query.question),
                semantic_ranker.rank(documents, model_input.query.question),
            )
        )
    else:
        raise ValueError("Unsupported evaluation profile")
    if profile == "source_aware_raw":
        parents = {source.source_id: source for source in model_input.history}
        unique = {}
        for index in order:
            match = documents[index]
            parent_id = match.source_id.split("/c", 1)[0]
            if parent_id in unique:
                continue
            parent = parents[parent_id]
            start, end = context_window(parent.content, match.start_offset, match.end_offset)
            unique[parent_id] = EvidenceChunk(
                source_id=parent_id,
                session_id=parent.session_id,
                timestamp=parent.timestamp,
                role=parent.role,
                content=parent.content[start:end],
                start_offset=start,
                end_offset=end,
                truncated=start > 0 or end < len(parent.content),
            )
        documents = list(unique.values())
        order = list(range(len(documents)))
    selected = []
    for index in order:
        candidate = payload.model_copy(
            update={
                "sources": tuple(
                    sorted(
                        (documents[i] for i in [*selected, index]),
                        key=lambda s: (
                            _timestamp(s.timestamp),
                            s.session_id,
                            s.source_id,
                            s.start_offset,
                        ),
                    )
                )
            }
        )
        if len(candidate.model_dump_json().encode()) > budget_bytes:
            continue
        selected.append(index)
        payload = candidate
        if len(selected) == 12:
            break
    return payload


def _usage(usage):
    return usage.model_dump() if usage else None


def _wilson(correct, total):
    if not total:
        return None
    p, z = correct / total, 1.96
    center = (p + z * z / (2 * total)) / (1 + z * z / total)
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / (1 + z * z / total)
    return [max(0, center - half), min(1, center + half)]


async def run_suite(
    path: Path,
    reader,
    *,
    judge=None,
    profiles=("recent_window", "lexical_raw", "full_history"),
    limit=5,
    seed=17,
    budget_bytes=12000,
    model="unspecified",
    dataset_condition="unspecified",
    progress=None,
    dataset_metadata=None,
    semantic_ranker=None,
    provider_config=None,
    retrieval_only=False,
):
    if (
        not profiles
        or len(set(profiles)) != len(profiles)
        or any(
            name
            not in {
                "recent_window",
                "lexical_raw",
                "source_aware_raw",
                "hybrid_raw",
                "full_history",
            }
            for name in profiles
        )
    ):
        raise ValueError("Invalid evaluation profiles")
    if "hybrid_raw" in profiles and semantic_ranker is None:
        raise ValueError("Hybrid profile requires a local semantic ranker")
    if not 512 <= budget_bytes <= 24000:
        raise ValueError("E1 reader budget must be 512..24000 UTF-8 bytes")
    if retrieval_only:
        judge = None
    cases, checksum, source_count = await asyncio.to_thread(load_cases, path, limit, seed)
    if dataset_metadata and dataset_metadata.get("sha256") != checksum:
        raise ValueError("Dataset manifest hash mismatch")
    report = {
        "schema_version": "hmph-e1-v1",
        "run_status": "running",
        "mode": "retrieval-only" if retrieval_only else "inference",
        "created_at": datetime.now(UTC).isoformat(),
        "code": await asyncio.to_thread(_code_version),
        "dataset_sha256": checksum,
        "dataset_condition": dataset_condition,
        "dataset_case_count": source_count,
        "dataset": {
            k: dataset_metadata[k]
            for k in ("name", "revision", "license", "source_url", "sha256")
            if dataset_metadata and k in dataset_metadata
        },
        "case_count": len(cases),
        "selection": {
            "seed": seed,
            "method": "smallest_sha256_of_seed_and_id",
            "ids": [case.case_id for case in cases],
        },
        "reader_model": model,
        "provider_config": provider_config,
        "retrieval_config": {
            "version": "bm25-word-cjk-stop-v2",
            "chunk_chars": 400,
            "overlap_chars": 80,
            "bm25_k1": 1.2,
            "bm25_b": 0.75,
            "rrf_k": 60,
            "candidate_limit": 80,
            "source_aware": {"max_chars_per_message": 800, "unique_parent_messages": True},
        },
        "judge_model": model if judge else None,
        "reader_budget_bytes": budget_bytes,
        "max_source_chunks": 12,
        "writer": "not_run",
        "planning_model": "not_run",
        "embedding": semantic_ranker.metadata if semantic_ranker else "not_run",
        "guard": "not_run_evaluation_only",
        "official_score": False,
        "frontier_status": "not_evaluated",
        "inference_requests": 0,
        "profiles": {},
        "predictions": [],
        "limitations": [
            "Development sample; no held-out or frontier noninferiority claim.",
            "Reader and optional judge share a model; this is not the official GPT-4o judge.",
            "Raw baselines omit B0 rolling summary and H-memory; writer is not run.",
            "Tokens are lower bounds if failed calls lack usage; prices not supplied.",
            "Public benchmark pretraining contamination cannot be ruled out.",
            "Wilson intervals are descriptive; no clustered paired comparison is claimed.",
        ],
    }
    for profile in profiles:
        records, gold_total, hit_total = [], 0, 0
        gold_messages, hit_messages = 0, 0
        for case in cases:
            started = time.monotonic()
            record = {
                "case_id": case.case_id,
                "profile": profile,
                "category": case.gold.category,
                "abstention_expected": case.gold.abstention,
                "status": "unsupported",
                "answer": None,
                "correct": None,
                "reader_usage": None,
                "judge_usage": None,
                "reader_attempted": False,
                "judge_attempted": False,
                "error_code": None,
            }
            report["predictions"].append(record)
            try:
                exposed = await asyncio.to_thread(
                    prepare_input, case.model_input(), profile, budget_bytes, semantic_ranker
                )
                record["preparation_seconds"] = round(time.monotonic() - started, 3)
                if profile == "hybrid_raw":
                    record["embedding_phases"] = getattr(semantic_ranker, "last_metrics", None)
                if exposed is not None:
                    selected = {source.session_id for source in exposed.sources}
                    gold_sessions = set(case.gold.evidence_sessions)
                    selected_messages = {
                        source.source_id.split("/c", 1)[0] for source in exposed.sources
                    }
                    message_gold = set(case.gold.evidence_messages)
                    gold_messages += len(message_gold)
                    hit_messages += len(selected_messages & message_gold)
                    hit_total += len(selected & gold_sessions)
                    gold_total += len(gold_sessions)
                    record.update(
                        selected_source_ids=[s.source_id for s in exposed.sources],
                        selected_sessions=sorted(selected),
                        gold_evidence_sessions=len(gold_sessions),
                        evidence_session_hits=len(selected & gold_sessions),
                        gold_evidence_messages=len(message_gold),
                        evidence_message_hits=len(selected_messages & message_gold),
                        payload_bytes=len(exposed.model_dump_json().encode()),
                        status="failed",
                    )
                    if retrieval_only:
                        record["status"] = "retrieved"
                    else:
                        record["reader_attempted"] = True
                        report["inference_requests"] += 1
                        if progress:
                            await progress(report)
                        phase_started = time.monotonic()
                        async with asyncio.timeout(45):
                            reply = await reader(exposed)
                        record.update(
                            reader_seconds=round(time.monotonic() - phase_started, 3),
                            answer=reply.answer,
                            abstained=reply.abstained,
                            reader_usage=_usage(reply.usage),
                            source_reference_errors=len(
                                set(reply.source_ids) - {s.source_id for s in exposed.sources}
                            ),
                            status="completed",
                        )
                        if judge:
                            record["judge_attempted"] = True
                            report["inference_requests"] += 1
                            if progress:
                                await progress(report)
                            phase_started = time.monotonic()
                            async with asyncio.timeout(45):
                                grade = await judge(
                                    GradeInput(
                                        question=case.query.question,
                                        reference_answer=str(case.gold.answer),
                                        hypothesis=reply.answer,
                                        abstention_expected=case.gold.abstention,
                                    )
                                )
                            record.update(
                                correct=grade.correct,
                                judge_usage=_usage(grade.usage),
                                judge_seconds=round(time.monotonic() - phase_started, 3),
                            )
            except Exception:
                # Keep the case and attempt in the denominator without logging raw exceptions.
                record.update(
                    status="failed",
                    error_code="judge_failed" if record["judge_attempted"] else "reader_failed",
                )
            record["seconds"] = round(time.monotonic() - started, 3)
            records.append(record)
            if progress:
                await progress(report)
        completed = sum(r["status"] == "completed" for r in records)
        correct = sum(r["correct"] is True for r in records)
        attempts = sum(r[stage + "_attempted"] for r in records for stage in ("reader", "judge"))
        usages = [
            r[stage + "_usage"]
            for r in records
            for stage in ("reader", "judge")
            if r[stage + "_usage"]
        ]
        report["profiles"][profile] = {
            "total": len(records),
            "completed": completed,
            "retrieved": sum(r["status"] == "retrieved" for r in records),
            "failed": sum(r["status"] == "failed" for r in records),
            "unsupported": sum(r["status"] == "unsupported" for r in records),
            "accuracy": correct / len(records)
            if judge and any(r["reader_attempted"] for r in records)
            else None,
            "accuracy_denominator": "all selected cases; unsupported cases remain unpassed",
            "accuracy_wilson_95": _wilson(correct, len(records)) if judge and attempts else None,
            "evidence_session_recall": hit_total / gold_total if gold_total else None,
            "evidence_message_recall": hit_messages / gold_messages if gold_messages else None,
            "evidence_scope": "supported_inputs; selected source unit, not exact supporting text",
            "inference_attempts": attempts,
            "usage_complete": len(usages) == attempts,
            "reported_tokens": sum(v["total_tokens"] for v in usages),
            "seconds_all_cases": sum(r["seconds"] for r in records),
            "phase_seconds": {
                stage: sum(r.get(stage + "_seconds", 0) for r in records)
                for stage in ("preparation", "reader", "judge")
            },
        }
    report["run_status"] = "completed"
    return report


def save_report(path: Path, report):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    # Windows indexers can briefly hold the previous report without delete sharing.
    for attempt in range(7):
        try:
            temporary.replace(path)
            return
        except PermissionError:
            if attempt == 6:
                raise
            time.sleep(0.02 * 2**attempt)
