"""Posthoc source-only reader control on the SAME saved retrieval snapshots.

No writer/retrieval rerun and no replacement of the frozen primary predictions.
Additional reader calls share the original SQLite run and per-case hard limits.
"""

import argparse
import copy
import hashlib
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from selection_profile import read_answer, save  # noqa: E402

from hmph.eval_budget import RunBudget  # noqa: E402
from hmph.selection_diagnostics import corrected_cases  # noqa: E402
from hmph.selection_eval import grade, load_selection, screen_profiles  # noqa: E402


def run(directory):
    from dotenv import dotenv_values
    from pydantic import SecretStr

    original = (directory / "report.json").read_bytes()
    primary = json.loads(original)
    plan = primary["plan"]
    for name, expected in plan["code_hashes"].items():
        if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != expected:
            raise ValueError("Frozen code changed")
    cases, checksum = load_selection(ROOT / "evals/fixtures/framework-selection-dev.json")
    errata = json.loads(
        (ROOT / "evals/fixtures/framework-selection-source-errata-v1.json").read_text(
            encoding="utf-8"
        )
    )
    cases = corrected_cases(cases, checksum, errata)
    values = {**dotenv_values(ROOT / ".env.local"), **os.environ}
    secret = values["HMPH_DEEPSEEK_API_KEY"]
    key = SecretStr(secret)
    budget = RunBudget(directory / "calls.sqlite", **plan["limits"])
    before = {entry["reservation"] for entry in budget.snapshot()["calls"]}
    profiles = copy.deepcopy(primary["profiles"])
    # Uniformly apply the independently audited label correction to the unchanged baseline.
    for case, row in zip(cases, profiles["raw"]["cases"], strict=True):
        from hmph.selection_eval import Reply

        row["scores"] = grade(
            case,
            Reply.model_validate_json(json.dumps(row["reply"])),
            {sid for item in row["evidence"] for sid in item["source_ids"]},
        )
    output = directory / "diagnostics-source-only.json"
    if output.exists():
        raise ValueError("Do not overwrite diagnostic predictions")
    report = {
        "kind": "posthoc_sources_only_control",
        "status": "running",
        "primary_report_sha256": hashlib.sha256(original).hexdigest(),
        "dataset_sha256": checksum,
        "source_errata": errata,
        "profiles": {},
        "writer_and_retrieval_reused": True,
        "primary_predictions_unchanged": True,
        "limitation": "Known development trajectories; this is not a frozen acceptance result.",
    }
    save(output, report, secret)
    for profile in ("mem0", "graphiti"):
        rows = []
        report["profiles"][profile] = {"status": "running", "cases": rows}
        for case, old in zip(cases, primary["profiles"][profile]["cases"], strict=True):
            assert case.case_id == old["case_id"]
            row = {
                "case_id": case.case_id,
                "category": case.category,
                "status": "failed",
                "timings": {},
                "cleanup": copy.deepcopy(old["cleanup"]),
                "scope_isolation": old.get("scope_isolation", False),
            }
            started = time.monotonic()
            try:
                if old["status"] != "completed":
                    raise ValueError("No completed original retrieval snapshot")
                evidence = [item for item in old["evidence"] if item["kind"] == "source"]
                reply, ids, _ = read_answer(
                    case.query, evidence, key, plan["model"], budget, case.case_id, profile, row
                )
                row.update(status="completed", scores=grade(case, reply, ids))
            except Exception as error:
                row["error_type"] = type(error).__name__
            row["seconds"] = time.monotonic() - started
            rows.append(row)
            save(output, report, secret)
        report["profiles"][profile]["status"] = (
            "completed"
            if all(row["status"] == "completed" for row in rows)
            else "completed_with_failures"
        )
        profiles[profile] = report["profiles"][profile]
    report["screening"] = screen_profiles(profiles, cases)
    report["additional_calls"] = [
        entry for entry in budget.snapshot()["calls"] if entry["reservation"] not in before
    ]
    report["status"] = (
        "completed"
        if all(value["status"] == "completed" for value in report["profiles"].values())
        else "incomplete"
    )
    if (directory / "report.json").read_bytes() != original:
        raise RuntimeError("Primary report changed")
    save(output, report, secret)
    print(
        json.dumps(
            {
                "status": report["status"],
                "extra_calls": len(report["additional_calls"]),
                "screening": report["screening"],
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", type=Path, required=True)
    args = parser.parse_args()
    run(args.directory.resolve())
