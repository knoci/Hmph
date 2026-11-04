"""Frozen, bounded TiMem component pilot. Preparation does not call a model."""

import argparse
import asyncio
import contextlib
import hashlib
import io
import json
import logging
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hmph.eval_budget import RunBudget  # noqa: E402
from hmph.timem_reference import (  # noqa: E402
    TIMEM_REVISION,
    LocalTiMem,
    MeteredTiMemLLM,
    install_provider,
    leaf_sources,
    local_test_key,
)

HISTORY = [
    {
        "id": "s01",
        "at": "2026-10-05T19:00:00",
        "messages": [
            {"id": "m01", "text": "I work at Cedar Studio. I began learning the cello yesterday."},
            {"id": "m02", "text": "My teacher is Anika. My cello lessons are on Tuesdays."},
        ],
    },
    {
        "id": "s02",
        "at": "2026-10-12T19:00:00",
        "messages": [
            {
                "id": "m03",
                "text": "I left Cedar Studio and now work at Maple Lab. My teacher is still Anika.",
            },
            {
                "id": "m04",
                "text": "My cello lesson moved from Tuesday to Thursday, starting this week.",
            },
        ],
    },
]
QUESTIONS = [
    {"question": "Where do I currently work?", "answer": "Maple Lab", "sources": ["m03"]},
    {
        "question": "On which day are my cello lessons now?",
        "answer": "Thursday",
        "sources": ["m04"],
    },
]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, value):
    temporary = path.with_suffix(".partial")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


async def execute(directory, plan):
    budget = RunBudget(directory / "calls.sqlite", **plan["limits"])
    llm = MeteredTiMemLLM(
        local_test_key(ROOT),
        budget,
        "case-001",
        max_tokens=plan["max_output_tokens"],
        stop_file=directory / "STOP",
    )
    native, report = LocalTiMem(), {"status": "running", "rows": [], "layers": {}}
    save(directory / "report.json", report)
    try:
        install_provider(llm)
        await native.open(directory)
        sessions = []
        for session in plan["history"]:
            when = datetime.fromisoformat(session["at"])
            text = "\n".join(m["id"] + ": " + m["text"] for m in session["messages"])
            content = await native.generator.generate_l1_content(text)
            leaf = await native.add(
                content,
                level="L1",
                when=when,
                session_id=session["id"],
                sources=[m["id"] for m in session["messages"]],
            )
            content = await native.generator.generate_l2_content([leaf.content])
            sessions.append(
                await native.add(
                    content, level="L2", when=when, session_id=session["id"], children=[leaf.id]
                )
            )
        when = datetime.fromisoformat(plan["history"][-1]["at"])
        weeks = []
        for session in sessions:
            date = session.time_window_start
            content = await native.generator.generate_l3_content(
                [session.content], date=date.date().isoformat()
            )
            daily = await native.add(content, level="L3", when=date, children=[session.id])
            content = await native.generator.generate_l4_content(
                [daily.content],
                year=2026,
                week_number=date.isocalendar().week,
                week_start=date.date().isoformat(),
                week_end=date.date().isoformat(),
            )
            weeks.append(await native.add(content, level="L4", when=date, children=[daily.id]))
        content = await native.generator.generate_l5_content(
            [week.content for week in weeks],
            year=2026,
            month=10,
            month_start="2026-10-01",
            month_end="2026-10-31",
        )
        monthly = await native.add(
            content, level="L5", when=when, children=[week.id for week in weeks]
        )
        report["layers"] = {
            level: sum(n.level.value == level for n in native.nodes.values())
            for level in ("L1", "L2", "L3", "L4", "L5")
        }
        report["top_level_sources"] = leaf_sources(monthly.id, native.nodes, native.leaf_map)
        # Deliberately no answer model: this pilot checks component retrieval and
        # leaf coverage. A full public-data comparison is a separate experiment.
        for question in plan["questions"]:
            ids, sources = await native.retrieve(question["question"])
            report["rows"].append(
                {
                    "question": question["question"],
                    "native_ids": ids,
                    "source_ids": sources,
                    "required_sources_present": set(question["sources"]).issubset(sources),
                }
            )
        report["status"] = "completed"
    except Exception as exc:
        from hmph.reference_transport import failure_code

        report.update(status="failed", error_type=type(exc).__name__, error_code=failure_code(exc))
        # Only static native/local exceptions, never provider error bodies.
        if not budget.snapshot()["calls"]:
            report["initialization_error"] = str(exc)[:250]
    finally:
        if getattr(native, "client", None):
            report["partial_layers"] = {
                level: sum(n.level.value == level for n in native.nodes.values())
                for level in ("L1", "L2", "L3", "L4", "L5")
            }
            report["collection_removed"] = native.close()
        await llm.close()
        report["budget"] = budget.snapshot()
        report["scope"] = (
            "component pilot; manually supplied layer batches; not full TiMem reproduction"
        )
        save(directory / "report.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    directory = args.directory.resolve()
    repo = ROOT / "artifacts/references/TiMem"
    observed = subprocess.check_output(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
    ).strip()
    if observed != TIMEM_REVISION:
        raise ValueError("TiMem revision mismatch")
    subprocess.run(["git", "-C", str(repo), "diff", "HEAD", "--quiet"], check=True)
    files = [
        ROOT / "src/hmph/timem_reference.py",
        Path(__file__),
        ROOT / "src/hmph/eval_budget.py",
        repo / "config/prompts.yaml",
        repo / "config/settings.yaml",
    ]
    hashes = {str(p.relative_to(ROOT)): digest(p) for p in files}
    if not args.run:
        directory.mkdir(parents=True, exist_ok=False)
        save(
            directory / "plan.json",
            {
                "revision": TIMEM_REVISION,
                "hashes": hashes,
                "history": HISTORY,
                "questions": QUESTIONS,
                "limits": {
                    "max_calls": 16,
                    "max_input_bytes": 2 * 1024 * 1024,
                    "per_case_calls": 16,
                },
                "model": "deepseek-flash",
                "max_output_tokens": 2048,
                "paid_retries": 0,
                "profile": (
                    "native generator L1-L5; embedded Qdrant; rule planner + L1/L2 retrieval"
                ),
                "stopping": "first exception, missing source mapping, STOP or budget exhaustion",
                "native_scheduler": False,
                "product_enabled": False,
            },
        )
        print("Prepared TiMem component pilot; model calls=0")
        return
    plan = json.loads((directory / "plan.json").read_text(encoding="utf-8"))
    if plan["hashes"] != hashes or (directory / "report.json").exists():
        raise ValueError("Frozen code/config changed or run already started")
    logging.disable(logging.CRITICAL)
    # Native code logs full synthetic content to stdout; suppress it completely.
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        report = asyncio.run(execute(directory, plan))
    print(
        json.dumps(
            {
                "status": report["status"],
                "layers": report["layers"],
                "rows": report["rows"],
                "calls": len(report["budget"]["calls"]),
                "collection_removed": report.get("collection_removed"),
                "error": report.get("initialization_error", report.get("error_code")),
            }
        )
    )


if __name__ == "__main__":
    main()
