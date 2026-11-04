"""One P1 native profile in its pinned environment. Only explicit --run spends calls."""

import argparse
import asyncio
import contextlib
import hashlib
import io
import json
import logging
import os
import subprocess
import sys
import time
from dataclasses import replace
from importlib.metadata import version
from pathlib import Path
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hmph.eval_budget import RunBudget  # noqa: E402
from hmph.selection_eval import (  # noqa: E402
    Reply,
    Source,
    raw_context,
    reader_messages,
    source_text,
)
from hmph.trajectory_eval import (  # noqa: E402
    RunConfiguration,
    grade_trajectory,
    load_suite,
    source_context,
)


def save(path, data, secret):
    raw = json.dumps(data, ensure_ascii=False, indent=2)
    if secret and secret in raw:
        raise RuntimeError("Credential output rejected")
    temporary = path.with_suffix(".partial")
    temporary.write_text(raw, encoding="utf-8")
    temporary.replace(path)


def inspect(profile):
    from hmph.graphiti_reference import GRAPHITI_REVISION
    from hmph.mem0_reference import MEM0_REVISION

    packages = (
        ("openai", "tzdata")
        if profile == "raw"
        else (
            ("mem0ai", "fastembed", "psycopg", "openai", "tzdata")
            if profile == "mem0"
            else ("graphiti-core", "fastembed", "neo4j", "openai", "tzdata")
        )
    )
    if profile != "raw":
        repo = ROOT / "artifacts/references" / ("mem0" if profile == "mem0" else "Graphiti")
        observed = subprocess.check_output(
            ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
        ).strip()
        if observed != (MEM0_REVISION if profile == "mem0" else GRAPHITI_REVISION):
            raise RuntimeError("Native revision mismatch")
        subprocess.run(["git", "-C", str(repo), "diff", "HEAD", "--quiet"], check=True)
    return {"profile": profile, "packages": {name: version(name) for name in packages}}


def transport(key, model, budget, case_id, profile, stage, *, validator=None, stop_file=None):
    from hmph.reference_transport import ReferenceTransport

    return ReferenceTransport(
        key,
        model=model,
        max_calls=budget.limits["per_case_calls"],
        max_tokens=256 if stage == "reader" else 1024,
        response_validator=validator,
        budget=budget.bind(case_id, profile, stage),
        stop_file=stop_file,
    )


def _read_answer(
    query, evidence, key, model, budget, case_id, profile, max_bytes=12000, *, stop_file=None
):
    import httpx
    from openai import OpenAI

    messages = reader_messages(query, evidence, max_bytes=max_bytes)
    supplied = json.loads(messages[1]["content"])["evidence"]
    ids = {source for item in supplied for source in item["source_ids"]}
    with httpx.Client(
        transport=transport(key, model, budget, case_id, profile, "reader", stop_file=stop_file),
        timeout=30,
        follow_redirects=False,
        trust_env=False,
    ) as http:
        sdk = OpenAI(
            api_key="native-placeholder",
            base_url="https://api.deepseek.com",
            http_client=http,
            max_retries=0,
        )
        result = sdk.chat.completions.create(
            model=model,
            max_tokens=256,
            response_format={"type": "json_object"},
            messages=messages,
            temperature=0,
        )
        reply = Reply.model_validate_json(result.choices[0].message.content)
    return reply, ids, supplied


def read_answer(
    query,
    evidence,
    key,
    model,
    budget,
    case_id,
    profile,
    row,
    *,
    case=None,
    evidence_mode="native",
    max_bytes=12000,
    stop_file=None,
):
    started = time.monotonic()
    try:
        if evidence_mode == "source_only":
            evidence = source_context(case, evidence)
        reply, ids, supplied = _read_answer(
            query, evidence, key, model, budget, case_id, profile, max_bytes, stop_file=stop_file
        )
        # Preserve predictions even if native cleanup subsequently fails.
        row.update(reply=reply.model_dump(mode="json"), evidence=supplied)
        return reply, ids, supplied
    finally:
        row["timings"]["reader_seconds"] = time.monotonic() - started


@contextlib.contextmanager
def mem0_context(
    history, query, key, model, budget, case_id, cleanup, database_url, timings, *, stop_file=None
):
    began = time.monotonic()
    import httpx
    from openai import OpenAI

    from hmph.mem0_backend import validate_extraction
    from hmph.mem0_reference import native_mem0

    directory = ROOT / "artifacts/references/runs" / ("p1-" + uuid4().hex)
    sources = {s.source_id: s for s in history}
    writer = transport(
        key,
        model,
        budget,
        case_id,
        "mem0",
        "writer",
        validator=validate_extraction,
        stop_file=stop_file,
    )
    with httpx.Client(
        transport=writer, timeout=30, trust_env=False, follow_redirects=False
    ) as http:
        sdk = OpenAI(
            api_key="native-placeholder",
            base_url="https://api.deepseek.com",
            http_client=http,
            max_retries=0,
        )
        with native_mem0(
            directory, sdk, model=model, database_url=database_url, cleanup_audit=cleanup
        ) as memory:
            scope = {"user_id": "synthetic_user"}
            for source in history:
                memory.add(
                    source_text(source),
                    **scope,
                    metadata={
                        "source_id": source.source_id,
                        "observed_at": source.observed_at.isoformat(),
                    },
                )
            timings["build_seconds"] = time.monotonic() - began
            began = time.monotonic()
            retrieved = memory.search(query.question, filters=scope, top_k=8, threshold=0)[
                "results"
            ]
            if memory.search(
                query.question, filters={"user_id": "another_user"}, top_k=8, threshold=0
            )["results"]:
                raise RuntimeError("Native scope isolation failed")
            evidence = []
            included = set()
            for hit in retrieved:
                source_id = (hit.get("metadata") or {}).get("source_id")
                if source_id not in sources:
                    raise RuntimeError("Unknown native source")
                evidence.append(
                    {
                        "kind": "fact",
                        "content": hit["memory"],
                        "source_ids": [source_id],
                        "provenance": "native_batch_metadata_incomplete_dependencies",
                    }
                )
                if source_id not in included:
                    source = sources[source_id]
                    evidence.append(
                        {
                            "kind": "source",
                            "content": source.content,
                            "observed_at": source.observed_at.isoformat(),
                            "source_ids": [source_id],
                            "provenance": "original_source",
                        }
                    )
                    included.add(source_id)
            timings["retrieval_and_scope_check_seconds"] = time.monotonic() - began
            yield evidence
    # The native context manager verifies PG and SQLite; no raw logs are written.


@contextlib.asynccontextmanager
async def graphiti_context(
    history,
    query,
    key,
    model,
    budget,
    case_id,
    cleanup,
    timings,
    *,
    stop_file=None,
    source_description="synthetic user statement",
    failure_observer=None,
):
    began = time.monotonic()
    import httpx
    from graphiti_core.nodes import EpisodeType
    from openai import AsyncOpenAI

    from hmph.graphiti_reference import graph_context, native_graphiti
    from hmph.reference_transport import AsyncReferenceTransport

    writer = AsyncReferenceTransport(
        key,
        model=model,
        max_calls=budget.limits["per_case_calls"],
        max_tokens=1024,
        budget=budget.bind(case_id, "graphiti", "writer"),
        stop_file=stop_file,
        failure_observer=failure_observer,
    )
    async with httpx.AsyncClient(
        transport=writer, timeout=30, trust_env=False, follow_redirects=False
    ) as http:
        sdk = AsyncOpenAI(
            api_key="native-placeholder",
            base_url="https://api.deepseek.com",
            http_client=http,
            max_retries=0,
        )
        async with native_graphiti(sdk, model=model, cleanup_audit=cleanup) as (graph, group):
            mapping, sources = {}, {}
            for source in history:
                body = source_text(source)
                result = await graph.add_episode(
                    name=source.source_id,
                    episode_body=body,
                    source_description=source_description,
                    reference_time=source.observed_at,
                    source=EpisodeType.text,
                    group_id=group,
                )
                mapping[result.episode.uuid] = source.source_id
                sources[source.source_id] = source
            timings["build_seconds"] = time.monotonic() - began
            began = time.monotonic()
            # Query timestamp is knowledge cutoff, not the event date asked about.
            # Every fixture source is already <= cutoff. Include closed historical edges
            # and let the common reader use their native valid/invalid times.
            retrieved = await graph_context(graph, group, query.question, None)
            if await graph.search(query.question, group_ids=[group + "_other"]):
                raise RuntimeError("Native scope isolation failed")
            evidence = []
            for item in retrieved:
                ids = [mapping[identifier] for identifier in item["source_ids"]]
                if item["kind"] == "source":
                    source = sources[ids[0]]
                    if item["content"] != source_text(source):
                        raise RuntimeError("Native source content mismatch")
                    evidence.append(
                        {
                            "kind": "source",
                            "content": source.content,
                            "observed_at": source.observed_at.isoformat(),
                            "source_ids": ids,
                            "provenance": "original_source",
                        }
                    )
                else:
                    evidence.append(
                        {
                            "kind": "fact",
                            "content": item["fact"],
                            "valid_at": item["valid_at"],
                            "invalid_at": item["invalid_at"],
                            "source_ids": ids,
                            "provenance": "native_episode_dependencies",
                        }
                    )
            timings["retrieval_and_scope_check_seconds"] = time.monotonic() - began
            yield evidence


async def run(args):
    from dotenv import dotenv_values
    from pydantic import SecretStr

    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    for name, expected in plan["code_hashes"].items():
        if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != expected:
            raise ValueError("Frozen code hash mismatch")
    config = RunConfiguration.model_validate(plan["configuration"])
    if config.limits != plan["limits"] or config.reader_bytes != plan["reader_messages_bytes"]:
        raise ValueError("Frozen configuration mismatch")
    cases, checksum = load_suite(args.suite, require_frozen=True)
    if config.case_ids:
        selected = {case.case_id: case for case in cases}
        cases = [selected[identifier] for identifier in config.case_ids]
    if [case.case_id for case in cases] != plan["case_ids"]:
        raise ValueError("Frozen case selection mismatch")
    if plan["dataset_sha256"] != checksum:
        raise ValueError("Frozen dataset hash mismatch")
    values = {**dotenv_values(ROOT / ".env"), **dotenv_values(ROOT / ".env.local"), **os.environ}
    secret, model = values.get("HMPH_DEEPSEEK_API_KEY"), plan["model"]
    if not secret or model not in {"deepseek-flash", "deepseek-v4-pro"}:
        raise ValueError("Local DeepSeek configuration required")
    key = SecretStr(secret)
    limits = plan["limits"]
    budget = RunBudget(args.plan.parent / "calls.sqlite", **limits)
    environment = inspect(args.profile)
    if environment != plan["environments"][args.profile]:
        raise ValueError("Frozen environment mismatch")
    report = {
        "profile": args.profile,
        "dataset_sha256": checksum,
        "cases": [],
        "status": "running",
        "environment": environment,
        "benchmark": False,
        "frontier_claim": False,
    }
    output = args.plan.parent / (args.profile + ".json")
    if output.exists():
        raise ValueError("Profile output already exists; do not silently rerun paid cases")
    save(output, report, secret)
    previous_logging = logging.root.manager.disable
    try:
        logging.disable(logging.CRITICAL)
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            for case in cases:
                row = {
                    "case_id": case.case_id,
                    "category": case.category,
                    "status": "failed",
                    "cleanup": {},
                    "timings": {},
                }
                started = time.monotonic()
                try:
                    history = tuple(
                        Source.model_validate(s) for s in case.writer_input()["history"]
                    )
                    if args.profile == "raw":
                        before = time.monotonic()
                        evidence = raw_context(replace(case, history=history))
                        row["timings"]["retrieval_and_scope_check_seconds"] = (
                            time.monotonic() - before
                        )
                        row["cleanup"]["not_applicable"] = True
                        reply, ids, selected = read_answer(
                            case.query,
                            evidence,
                            key,
                            model,
                            budget,
                            case.case_id,
                            args.profile,
                            row,
                            case=case,
                            evidence_mode=config.evidence_mode,
                            max_bytes=config.reader_bytes,
                        )
                    elif args.profile == "mem0":
                        with mem0_context(
                            history,
                            case.query,
                            key,
                            model,
                            budget,
                            case.case_id,
                            row["cleanup"],
                            values.get("HMPH_DATABASE_URL"),
                            row["timings"],
                        ) as evidence:
                            reply, ids, selected = read_answer(
                                case.query,
                                evidence,
                                key,
                                model,
                                budget,
                                case.case_id,
                                args.profile,
                                row,
                                case=case,
                                evidence_mode=config.evidence_mode,
                                max_bytes=config.reader_bytes,
                            )
                    else:
                        async with graphiti_context(
                            history,
                            case.query,
                            key,
                            model,
                            budget,
                            case.case_id,
                            row["cleanup"],
                            row["timings"],
                        ) as evidence:
                            reply, ids, selected = await asyncio.to_thread(
                                read_answer,
                                case.query,
                                evidence,
                                key,
                                model,
                                budget,
                                case.case_id,
                                args.profile,
                                row,
                                case=case,
                                evidence_mode=config.evidence_mode,
                                max_bytes=config.reader_bytes,
                            )
                    row.update(
                        status="completed",
                        reply=reply.model_dump(mode="json"),
                        evidence=selected,
                        scores=grade_trajectory(case, reply, ids),
                        scope_isolation=True,
                    )
                except Exception as error:
                    row["error_type"] = type(error).__name__
                row["seconds"] = round(time.monotonic() - started, 3)
                report["cases"].append(row)
                save(output, report, secret)
                snapshot = budget.snapshot()
                if len(snapshot["calls"]) >= limits["max_calls"]:
                    break
        report["status"] = (
            "completed"
            if len(report["cases"]) == len(cases)
            and all(row["status"] == "completed" for row in report["cases"])
            else "completed_with_failures"
            if len(report["cases"]) == len(cases)
            else "budget_stopped"
        )
    finally:
        logging.disable(previous_logging)
        save(output, report, secret)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", choices=("raw", "mem0", "graphiti"), required=True)
    parser.add_argument("--inspect", action="store_true")
    parser.add_argument("--suite", type=Path)
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    if args.inspect:
        if args.suite:
            load_suite(args.suite, require_frozen=True)
        print(json.dumps(inspect(args.profile)))
    elif args.run and args.plan and args.suite:
        result = asyncio.run(run(args))
        print(
            json.dumps(
                {
                    "profile": args.profile,
                    "status": result["status"],
                    "completed": sum(row["status"] == "completed" for row in result["cases"]),
                }
            )
        )
    else:
        parser.error("Use --inspect or explicit --run --plan --suite")
