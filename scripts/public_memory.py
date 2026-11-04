"""P4-e bounded public development using existing native writers and common reader.

No product/default switch, no formal noninferiority conclusion, no old paid replay.
"""

import argparse
import asyncio
import contextlib
import hashlib
import io
import json
import logging
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from selection_profile import graphiti_context, mem0_context, read_answer  # noqa: E402
from timem_public import JUDGE, LME, LOCOMO, infer  # noqa: E402
from timem_run import digest, save  # noqa: E402

from hmph.eval_budget import RunBudget  # noqa: E402
from hmph.eval_runner import _timestamp, load_cases  # noqa: E402
from hmph.reference_transport import failure_code  # noqa: E402
from hmph.selection_eval import Gold, Query, SelectionCase, Source, raw_context  # noqa: E402
from hmph.timem_reference import local_test_key  # noqa: E402


def select_data():
    # First ranked LME history and LoCoMo record 0 were used in old paid studies.
    selected, _, _ = load_cases(LME, limit=2, seed=20261104)
    case = selected[1]
    lme = {
        "dataset": "LongMemEval",
        "original_id": case.case_id,
        "history": [m.model_dump() for m in case.history],
        "question": case.query.question,
        "timestamp": case.query.timestamp,
        "answer": case.gold.answer,
        "sources": list(case.gold.evidence_messages),
    }
    records = json.loads(LOCOMO.read_text(encoding="utf-8"))
    candidates = records[1:]
    record = min(candidates, key=lambda r: hashlib.sha256(str(r["sample_id"]).encode()).hexdigest())
    history, conversation = [], record["conversation"]
    for number in range(1, 100):
        if f"session_{number}" not in conversation:
            break
        timestamp = datetime.strptime(
            conversation[f"session_{number}_date_time"], "%I:%M %p on %d %B, %Y"
        ).isoformat()
        for item in conversation[f"session_{number}"]:
            history.append(
                {
                    "source_id": item["dia_id"],
                    "session_id": f"s{number:03}",
                    "timestamp": timestamp,
                    "role": "user",
                    "content": item["speaker"] + ": " + item["text"],
                }
            )
    question = next(q for q in record["qa"] if "answer" in q)
    return [
        lme,
        {
            "dataset": "LoCoMo",
            "original_id": record["sample_id"],
            "history": history,
            "question": question["question"],
            "timestamp": None,
            "answer": question["answer"],
            "sources": question["evidence"],
        },
    ]


def pack(record):
    """Chronological episode batches; preserve every message, speaker and source ID."""
    batches, mapping, buffer, leaves, session, observed = [], {}, [], [], None, None

    def flush():
        if not buffer:
            return
        identifier = f"s{len(batches) + 1:03}"
        batches.append(
            Source(source_id=identifier, observed_at=observed, content="\n".join(buffer))
        )
        mapping[identifier] = list(dict.fromkeys(leaves))
        buffer.clear()
        leaves.clear()

    for message in record["history"]:
        stamp = datetime(*_timestamp(message["timestamp"]), tzinfo=UTC)
        content = f"[{message['source_id']}] {message['role']}: {message['content']}"
        # Character slices preserve all text; 700 Unicode characters fit the byte admission.
        for start in range(0, len(content), 700):
            part = content[start : start + 700]
            if buffer and (
                session != message["session_id"]
                or len(("\n".join(buffer + [part])).encode()) > 3000
            ):
                flush()
            session, observed = message["session_id"], stamp
            buffer.append(part)
            leaves.append(message["source_id"])
    flush()
    query_time = (
        datetime(*_timestamp(record["timestamp"]), tzinfo=UTC)
        if record["timestamp"]
        else max(b.observed_at for b in batches)
    )
    required = [key for key, values in mapping.items() if set(values) & set(record["sources"])]
    case = SelectionCase(
        "",
        record["dataset"],
        tuple(batches),
        Query(question=record["question"], timestamp=query_time),
        Gold(
            answer=str(record["answer"]),
            aliases=(),
            evidence_sources=tuple(required),
            rationale="scoring only",
        ),
    )
    return case, mapping


def prepare(directory):
    directory.mkdir(parents=True, exist_ok=False)
    records = select_data()
    files = [
        Path(__file__),
        ROOT / "scripts/selection_profile.py",
        ROOT / "scripts/timem_public.py",
        *ROOT.glob("src/hmph/**/*.py"),
    ]
    plan = {
        "purpose": "P4-e two new full-history development examples, not formal acceptance",
        "prepared_at": datetime.now(UTC).isoformat(),
        "model": "deepseek-flash",
        "profiles": ["raw", "mem0", "graphiti"],
        "records": records,
        "data_hashes": {"LongMemEval": digest(LME), "LoCoMo": digest(LOCOMO)},
        "code_hashes": {str(p.relative_to(ROOT)): digest(p) for p in files},
        "limits": {"max_calls": 4096, "max_input_bytes": 128 * 1024 * 1024, "per_case_calls": 2048},
        "native_revisions": {
            "graphiti": "16bf514380de4e44884aed6779bece220e6fe470",
            "mem0": "753496cf1435d76c3a711f3af51303f7fdcb6420",
        },
        "reader_bytes": 12000,
        "episode_batch_bytes": 3000,
        "database": "postgresql://hmph:hmph_local_only@127.0.0.1:15439/hmph",
        "stopping": "first technical failure, STOP or budget exhaustion; no retry/resume/top-up",
        "limitations": [
            "n=1 history/question per dataset; no CI or paper acceptance",
            "E1 previously exposed all LME retrieval labels; no pristine holdout claim",
            "TiMem full scheduler not reproduced; mem0 fallback reuses pinned author engine",
            "chronological episode batches differ from per-message controlled product mode",
            "same-model judge bias; episode source expansion is not precise dependency proof",
            "raw is source-reader baseline, not Hmph product or all-three-memory ablation",
            "fixed currency price and latency gate not registered; monetary proximity unproven",
        ],
    }
    save(directory / "plan.json", plan)
    print(
        json.dumps(
            {
                "prepared": True,
                "histories": [
                    {
                        "dataset": r["dataset"],
                        "messages": len(r["history"]),
                        "bytes": sum(len(m["content"].encode()) for m in r["history"]),
                        "episode_batches": len(pack(r)[0].history),
                    }
                    for r in records
                ],
            }
        )
    )


async def run_profile(directory, profile):
    import hmph.graphiti_reference as graph_reference

    plan = json.loads((directory / "plan.json").read_text(encoding="utf-8"))
    if any(digest(ROOT / name) != expected for name, expected in plan["code_hashes"].items()):
        raise ValueError("Frozen source changed")
    output = directory / f"{profile}.json"
    if output.exists():
        raise ValueError("Profile already started; no paid resume")
    key = local_test_key(ROOT)
    budget = RunBudget(directory / "calls.sqlite", **plan["limits"])
    report = {"profile": profile, "status": "running", "rows": []}
    graph_reference.NEO4J_URI = "bolt://127.0.0.1:17688"
    graph_reference.NEO4J_PASSWORD = "hmph_graphiti_runtime_local_only"

    def checkpoint():
        save(output, {**report, "budget": budget.snapshot()})

    checkpoint()
    logging.disable(logging.CRITICAL)
    try:
        for index, record in enumerate(plan["records"], 1):
            if (directory / "STOP").exists():
                raise RuntimeError("research_stopped")
            case, mapping = pack(record)
            case_id = f"case-{index:03}"
            cleanup, timings = {}, {}
            row = {
                "dataset": record["dataset"],
                "status": "running",
                "timings": timings,
                "cleanup": cleanup,
                "episode_sources": mapping,
            }
            report["rows"].append(row)
            checkpoint()
            began = time.monotonic()
            with (
                contextlib.redirect_stdout(io.StringIO()),
                contextlib.redirect_stderr(io.StringIO()),
            ):
                if profile == "raw":
                    evidence = raw_context(case)
                    reply, allowed, supplied = read_answer(
                        case.query,
                        evidence,
                        key,
                        plan["model"],
                        budget,
                        case_id,
                        profile,
                        row,
                        stop_file=directory / "STOP",
                    )
                elif profile == "mem0":
                    with mem0_context(
                        case.history,
                        case.query,
                        key,
                        plan["model"],
                        budget,
                        case_id,
                        cleanup,
                        plan["database"],
                        timings,
                        stop_file=directory / "STOP",
                    ) as evidence:
                        reply, allowed, supplied = read_answer(
                            case.query,
                            evidence,
                            key,
                            plan["model"],
                            budget,
                            case_id,
                            profile,
                            row,
                            stop_file=directory / "STOP",
                        )
                        checkpoint()
                else:
                    async with graphiti_context(
                        case.history,
                        case.query,
                        key,
                        plan["model"],
                        budget,
                        case_id,
                        cleanup,
                        timings,
                        stop_file=directory / "STOP",
                        source_description="public chronological dialogue episode",
                        failure_observer=lambda diagnostic: save(
                            directory / "graphiti-failed-call.json", diagnostic
                        ),
                    ) as evidence:
                        reply, allowed, supplied = read_answer(
                            case.query,
                            evidence,
                            key,
                            plan["model"],
                            budget,
                            case_id,
                            profile,
                            row,
                            stop_file=directory / "STOP",
                        )
                        checkpoint()
            row["total_build_query_reader_seconds"] = time.monotonic() - began
            row["cited_sources_valid"] = set(reply.source_ids).issubset(allowed)
            grade = infer(
                key,
                budget,
                case_id,
                profile,
                "judge",
                [
                    {"role": "system", "content": JUDGE},
                    {
                        "role": "user",
                        "content": json.dumps(
                            {
                                "question": record["question"],
                                "reference": record["answer"],
                                "hypothesis": reply.answer,
                            },
                            ensure_ascii=False,
                        ),
                    },
                ],
                stop_file=directory / "STOP",
            )
            if type(grade.get("correct")) is not bool:
                raise ValueError("Invalid grade")
            row["correct"] = grade["correct"] and row["cited_sources_valid"]
            row["required_episode_coverage"] = set(case.gold.evidence_sources).issubset(
                {s for item in supplied for s in item["source_ids"]}
            )
            row["status"] = "completed"
            checkpoint()
        report["status"] = "completed"
    except BaseException as error:
        report["status"] = "failed"
        report["error_code"] = failure_code(error)
    finally:
        checkpoint()
    print(
        json.dumps(
            {
                "profile": profile,
                "status": report["status"],
                "completed": sum(r["status"] == "completed" for r in report["rows"]),
            }
        )
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", required=True, type=Path)
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--profile", choices=("raw", "mem0", "graphiti"))
    args = parser.parse_args()
    if args.profile:
        if not args.run:
            parser.error("Profile requires explicit --run")
        asyncio.run(run_profile(args.directory, args.profile))
    elif not args.run:
        prepare(args.directory)
    else:
        interpreters = {
            "raw": "framework-env",
            "mem0": "framework-env",
            "graphiti": "graphiti-runtime-env",
        }
        for profile, environment in interpreters.items():
            subprocess.run(
                [
                    str(ROOT / f"artifacts/{environment}/Scripts/python.exe"),
                    str(Path(__file__)),
                    "--directory",
                    str(args.directory),
                    "--run",
                    "--profile",
                    profile,
                ],
                check=True,
            )
            report = json.loads((args.directory / f"{profile}.json").read_text(encoding="utf-8"))
            if report["status"] != "completed":
                break


if __name__ == "__main__":
    main()
