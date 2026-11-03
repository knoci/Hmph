"""Freeze a bounded development run; paid execution always requires --run."""

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from hmph.eval_budget import RunBudget  # noqa: E402
from hmph.selection_eval import (  # noqa: E402
    READER_PROMPT,
    SCORING_VERSION,
    screen_profiles,
)
from hmph.trajectory_eval import RunConfiguration, TrajectoryCase, load_suite  # noqa: E402


def write(path, value):
    temporary = path.with_suffix(".partial")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def main(args):
    config = RunConfiguration.for_protocol(
        args.protocol,
        **{
            key: tuple(value) if key in {"profiles", "case_ids"} else value
            for key in (
                "profiles",
                "case_ids",
                "max_calls",
                "max_input_bytes",
                "per_case_calls",
                "reader_bytes",
                "evidence_mode",
            )
            if (value := getattr(args, key)) is not None
        },
    )
    directory = args.directory.resolve()
    allowed = (ROOT / "artifacts/evals").resolve()
    if not directory.is_relative_to(allowed) or directory == allowed or directory.exists():
        raise ValueError("Use a new run directory under artifacts/evals")
    cases, checksum = load_suite(args.suite, require_frozen=True)
    if (config.protocol == "p2a") != isinstance(cases[0], TrajectoryCase):
        raise ValueError("Suite does not match the selected protocol")
    if config.case_ids:
        selected = {case.case_id: case for case in cases}
        if not set(config.case_ids).issubset(selected):
            raise ValueError("Unknown development trajectory")
        cases = [selected[identifier] for identifier in config.case_ids]
    interpreters = {
        "raw": ROOT / "artifacts/framework-env/Scripts/python.exe",
        "mem0": ROOT / "artifacts/framework-env/Scripts/python.exe",
        "graphiti": ROOT / "artifacts/graphiti-env/Scripts/python.exe",
    }
    environments = {}
    interpreters = {profile: interpreters[profile] for profile in config.profiles}
    for profile, interpreter in interpreters.items():
        inspected = subprocess.run(
            [
                str(interpreter),
                str(ROOT / "scripts/selection_profile.py"),
                "--profile",
                profile,
                "--inspect",
                "--suite",
                str(args.suite.resolve()),
            ],
            capture_output=True,
            text=True,
            check=True,
            cwd=ROOT,
        )
        environments[profile] = json.loads(inspected.stdout)
    if len({entry["packages"]["openai"] for entry in environments.values()}) != 1:
        raise ValueError("Reader SDK versions must match")
    directory.mkdir(parents=True)
    hashes = {
        path: hashlib.sha256((ROOT / path).read_bytes()).hexdigest()
        for path in (
            "scripts/selection_run.py",
            "scripts/selection_profile.py",
            "src/hmph/selection_eval.py",
            "src/hmph/eval_budget.py",
            "src/hmph/reference_transport.py",
            "src/hmph/graphiti_reference.py",
            "src/hmph/mem0_reference.py",
            "src/hmph/mem0_backend.py",
            "src/hmph/trajectory_eval.py",
        )
    }
    plan = {
        "schema_version": "hmph-selection-plan-v2",
        "configuration": config.model_dump(mode="json"),
        "created_at": datetime.now(UTC).isoformat(),
        "dataset_sha256": checksum,
        "case_ids": [case.case_id for case in cases],
        "profiles": list(interpreters),
        "model": "deepseek-flash",
        "limits": config.limits,
        "writer_max_output_tokens": 1024,
        "reader_max_output_tokens": 256,
        "reader_messages_bytes": config.reader_bytes,
        "reader_temperature": 0,
        "reader_prompt_sha256": hashlib.sha256(READER_PROMPT.encode()).hexdigest(),
        "scoring": SCORING_VERSION,
        "code_hashes": hashes,
        "environments": environments,
        "adaptations": [
            "Both native writers get explicit observation timestamps; raw keeps timestamps.",
            "Native Mem0 search with batch-source expansion; dependencies may be incomplete.",
            "Graphiti combined RRF/source expansion; historical edges included; no reranker.",
            "Source observations precede query cutoff; no bitemporal reconstruction.",
            "Frozen short-answer aliases; rechecks are diagnostics and cannot replace scores.",
        ],
        "development_screen": {
            "completed_cases_per_profile": 24,
            "cleanup_and_scope_failures_allowed": 0,
            "max_answer_deficit_vs_raw": 2,
            "minimum_correct_per_category": 2,
            "abstention_correct_required": 3,
            "meaning": "Provisional P2 candidate screening only; no statistical frontier claim.",
        },
        "benchmark": False,
        "frontier_claim": False,
        "currency_cost": None,
        "price_version": None,
    }
    if config.protocol == "p2a":
        plan["development_screen"] = None
        plan["scope"] = "Source-only development calibration, not product or holdout acceptance"
        plan["build_watermarks"] = {case.case_id: case.build_watermark for case in cases}
        plan["adaptations"] += [
            "P2 writers receive only sources at or before the frozen build watermark.",
            "Readers get canonical original sources plus the unprocessed cross-session delta.",
            "The delta overlay is evaluation-only; product coordination is not implemented.",
            "The JSON reader uses a 5000-byte total messages cap, not the product companion chain.",
        ]
    write(directory / "plan.json", plan)
    budget = RunBudget(directory / "calls.sqlite", **plan["limits"])
    report = {"status": "prepared", "profiles": {}, "plan": plan, "calls": budget.snapshot()}
    write(directory / "report.json", report)
    if not args.run:
        print(
            json.dumps(
                {
                    "status": "prepared",
                    "case_count": len(cases),
                    "max_calls": config.max_calls,
                    "protocol": config.protocol,
                }
            )
        )
        return
    report["status"] = "running"
    write(directory / "report.json", report)
    for profile, interpreter in interpreters.items():
        result = subprocess.run(
            [
                str(interpreter),
                str(ROOT / "scripts/selection_profile.py"),
                "--profile",
                profile,
                "--suite",
                str(args.suite.resolve()),
                "--plan",
                str(directory / "plan.json"),
                "--run",
            ],
            cwd=ROOT,
            capture_output=True,
            text=True,
        )
        output = directory / (profile + ".json")
        report["profiles"][profile] = (
            json.loads(output.read_text(encoding="utf-8"))
            if output.exists()
            else {
                "status": "process_failed",
                "exit_code": result.returncode,
            }
        )
        report["calls"] = budget.snapshot()
        write(directory / "report.json", report)
        print(
            json.dumps(
                {
                    "profile": profile,
                    "exit_code": result.returncode,
                    "calls_so_far": len(report["calls"]["calls"]),
                }
            ),
            flush=True,
        )
    report["status"] = (
        "completed"
        if all(value["status"] == "completed" for value in report["profiles"].values())
        else "incomplete"
    )
    summary = {}
    for profile, value in report["profiles"].items():
        rows = value.get("cases", [])
        done = [row for row in rows if row["status"] == "completed"]
        summary[profile] = {
            "attempted_cases": len(rows),
            "completed": len(done),
            "answer_correct": sum(row["scores"]["answer_correct"] for row in done),
            "citation_valid": sum(row["scores"]["citation_valid"] for row in done),
            "correct": sum(row["scores"]["correct"] for row in done),
            "per_category": {
                category: sum(
                    row["scores"]["answer_correct"] for row in done if row["category"] == category
                )
                for category in sorted({case.category for case in cases})
            },
        }
    report["summary"] = summary
    report["screening"] = (
        screen_profiles(report["profiles"], cases)
        if config.protocol == "p1"
        else {
            "decision": "not_applicable",
            "meaning": "P2-a development calibration only; no eligibility decision.",
        }
    )
    calls = report["calls"]["calls"]
    report["cost"] = {
        profile: {
            stage: {
                "attempted_calls": len(
                    group := [
                        entry
                        for entry in calls
                        if entry["profile"] == profile and entry["stage"] == stage
                    ]
                ),
                "reported_known_tokens": sum(
                    (entry["record"].get("usage") or {}).get("total_tokens", 0) for entry in group
                ),
                "usage_complete": all(entry["record"].get("usage") is not None for entry in group),
                "input_bytes": sum(entry["record"]["input_bytes"] for entry in group),
            }
            for stage in ("writer", "reader")
        }
        for profile in interpreters
    }
    write(directory / "report.json", report)
    print(json.dumps({"status": report["status"], "summary": summary}, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument(
        "--suite", type=Path, default=ROOT / "evals/fixtures/framework-selection-dev.json"
    )
    parser.add_argument("--run", action="store_true", help="Explicitly enable bounded paid calls")
    parser.add_argument("--protocol", choices=("p1", "p2a"), default="p1")
    parser.add_argument("--profiles", nargs="+", choices=("raw", "mem0", "graphiti"))
    parser.add_argument("--case-ids", nargs="+")
    parser.add_argument("--max-calls", type=int)
    parser.add_argument("--max-input-bytes", type=int)
    parser.add_argument("--per-case-calls", type=int)
    parser.add_argument("--reader-bytes", type=int)
    parser.add_argument("--evidence-mode", choices=("native", "source_only"))
    main(parser.parse_args())
