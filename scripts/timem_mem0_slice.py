"""Run the frozen LoCoMo TiMem + mem0 comparison in an isolated database schema."""

import argparse
import asyncio
import hashlib
import json
import os
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from uuid import UUID, uuid4

from product_run import CapturingConversation, Services, evaluation_error_code, save
from public_product import import_history
from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import async_sessionmaker

from hmph.database import Database
from hmph.deepseek import PERSONA
from hmph.eval_budget import RunBudget
from hmph.framework_worker import FrameworkWorker
from hmph.mem0_backend import Mem0Backend, Mem0SourceReader
from hmph.memory import Memory
from hmph.memory_builds import MemoryBuilds
from hmph.models import MemoryBuild, MemorySpace
from hmph.online_memory import MemorySelection
from hmph.product_eval import RunControl
from hmph.selection_eval import Source
from hmph.shared_threads import SharedThreads
from hmph.timem_reference import (
    LocalTiMem,
    MeteredTiMemLLM,
    TiMemSupplement,
    build_snapshot_layers,
    install_provider,
    local_test_key,
)
from hmph.worker import Worker

ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / "artifacts/evals/p4e-public-native-dev-v3/timem-mem0-slice.json"
TIMEM = ROOT / "artifacts/references/TiMem"


def digest(data):
    return hashlib.sha256(data).hexdigest()


def load_history(plan):
    raw = (ROOT / plan["reference"]).read_bytes()
    if digest(raw) != plan["reference_sha256"]:
        raise ValueError("Frozen LoCoMo source changed")
    record = next(r for r in json.loads(raw)["records"] if r["dataset"] == "LoCoMo")
    additions = {row["after"]: row for row in plan["history"]["fixture_additions"]}
    ordered = []
    for row in record["history"]:
        observed = datetime.fromisoformat(row["timestamp"]).replace(tzinfo=UTC)
        ordered.append((row["source_id"], observed, row["content"]))
        if extra := additions.get(row["source_id"]):
            ordered.append((extra["source_id"], observed, extra["content"]))
    sources = tuple(
        Source(source_id=f"s{i:03}", observed_at=observed, content=body)
        for i, (_, observed, body) in enumerate(ordered, 1)
    )
    return SimpleNamespace(history=sources), {
        source.source_id: row[0] for source, row in zip(sources, ordered, strict=True)
    }


async def answer(db, identity, chat_id, reader, question, services, sources):
    # Roll back the first arm's conversation effects so the second arm has
    # identical recent turns, read watermark, controls and application state.
    async with db.engine.connect() as connection, connection.begin() as transaction:
        paired_db = SimpleNamespace(
            sessions=async_sessionmaker(
                connection, expire_on_commit=False, join_transaction_mode="create_savepoint"
            )
        )
        conversation = CapturingConversation(
            paired_db, digest_key=b"timem-mem0-slice", turn_timeout=45, source_reader=reader
        )
        if chat_id is None:
            chat_id = (
                await conversation.create_chat(
                    identity.owner_id, identity.character_id, uuid4().hex
                )
            ).id
        started = time.monotonic()
        accepted = await conversation.accept(
            identity.owner_id, chat_id, uuid4().hex, question, model_mode="deepseek"
        )
        lease = await conversation.claim("slice-reader", 60, model_mode="deepseek")
        if lease is None or lease.turn_id != accepted.id:
            raise RuntimeError("turn_not_claimed")
        worker = Worker(
            conversation,
            services.model("generation"),
            model_mode="deepseek",
            model_name="deepseek-flash",
        )
        request = await conversation.generation_request(lease)
        async with asyncio.timeout(45):
            # Same real publish path, with the experiment's single attempt.
            await worker._generate_reviewed(lease, request)
        turn = await conversation.get_turn(identity.owner_id, accepted.id)
        messages = await conversation.messages(identity.owner_id, chat_id)
        explanation = await conversation.memory_explanation(identity.owner_id, accepted.id)
        elapsed = time.monotonic() - started
        result = {
            "answer": "".join(
                m.content for m in messages if m.role == "assistant" and m.turn_id == accepted.id
            ),
            "sources": [
                sources.get(str(item.message_id), str(item.message_id))
                for item in request.raw_evidence
            ],
            "timem_sources": [
                sources.get(item, item)
                for item in (explanation["context"] or {}).get("timem_source_ids", [])
            ],
            "branch_status": (explanation["context"] or {}).get("branch_status", {}),
            "timem_strategy": (explanation["context"] or {}).get("timem_strategy"),
            "retrieval_status": request.workspace.retrieval_status,
            "generation_input": request.model_dump(mode="json"),
            "seconds": round(elapsed, 3),
            "usage": turn.usage,
            "status": turn.status,
        }
        await transaction.rollback()
        return result


async def execute(directory, plan):
    case, aliases = load_history(plan)
    key = local_test_key(ROOT)
    schema = "timem_mem0_" + uuid4().hex
    database = Database(os.environ["HMPH_SLICE_DATABASE_URL"], schema=schema)
    limits = plan["budget"]
    budget = RunBudget(
        directory / "calls.sqlite",
        max_calls=limits["maximum_calls"],
        max_input_bytes=limits["maximum_request_bytes"],
        per_case_calls=limits["timem_writer_max_calls"],
    )
    control = RunControl(directory / "STOP")
    backend = Mem0Backend(
        key,
        database_url=database.url,
        batch_bytes=3000,
        max_calls=limits["mem0_writer_max_calls"],
        budget=budget.bind("case-002", "mem0", "writer"),
        stop_file=control.stop_file,
    )
    builds = MemoryBuilds(database, digest_key=b"timem-mem0-slice")
    framework = FrameworkWorker(builds, backend)
    native = LocalTiMem()
    llm = MeteredTiMemLLM(key, budget, "case-001", stop_file=control.stop_file)
    services = {
        profile: Services(key, budget, "case-003", profile, control=control)
        for profile in plan["profiles"]
    }
    report = {"status": "running", "stage": "import", "rows": []}
    timem_closed = False

    def checkpoint():
        save(directory / "report.json", {**report, "budget": budget.snapshot()})

    checkpoint()
    try:
        async with database.engine.begin() as connection:
            await connection.execute(text(f'CREATE SCHEMA "{schema}"'))
        await database.migrate()
        conversation = CapturingConversation(database, digest_key=b"timem-mem0-slice")
        identity = await conversation.bootstrap()
        owner, character = identity.owner_id, identity.character_id
        chat = await conversation.create_chat(owner, character, "frozen-history")
        imported = await import_history(
            database, chat.id, case, format_source=lambda source: source.content
        )
        sources = {uuid: aliases[source_id] for uuid, source_id in imported.items()}
        thread_chat = await conversation.create_chat(owner, character, "thread-control")
        thread = await SharedThreads(conversation).write(
            owner,
            thread_chat.id,
            "seed-shared-plan",
            topic="visit the Saffron Observatory on Saturday",
            model_mode="deepseek",
        )
        thread_source = thread["source_message_id"]
        report["stage"] = "mem0_writer"
        checkpoint()
        requested = await builds.request(owner, character, "snapshot", backend="mem0")
        await framework.run_once(schedule_maintenance=False)
        build = await builds.get(owner, requested.id)
        if build.status != "ready":
            report["framework_error"] = build.error_code or build.status
            raise RuntimeError("mem0_build_failed")
        space = (await builds.spaces(owner, build.id))[0]
        report["mem0_build_id"] = str(build.id)
        report["mem0_usage"] = space.usage
        async with database.sessions() as session:
            frozen = await builds._sources(session, await session.get(MemoryBuild, build.id))
        await MemorySelection(database, digest_key=conversation.digest_key, allow_mem0=True).update(
            owner, character, "select", expected_revision=1, build_id=build.id
        )
        mem0 = Mem0SourceReader(
            database.url, model_cache=ROOT / "artifacts/models/multilingual-minilm"
        )
        report["stage"] = "timem_writer"
        checkpoint()
        install_provider(llm)
        await native.open(directory)
        await build_snapshot_layers(native, frozen)
        report["timem_layers"] = sorted({node.level.value for node in native.nodes.values()})
        readers = {
            "mem0_only": mem0,
            "mem0_timem": TiMemSupplement(mem0, native, build_id=build.id, space_id=space.id),
        }

        async def pair(probe, chat_id):
            for profile, reader in readers.items():
                control.check()
                if len(report["rows"]) >= limits["product_max_calls"]:
                    raise RuntimeError("product_call_budget")
                row = await answer(
                    database,
                    identity,
                    chat_id,
                    reader,
                    probe["question"],
                    services[profile],
                    sources,
                )
                missing = sorted(set(probe.get("required_sources", ())) - set(row["sources"]))
                report["rows"].append(
                    {
                        "probe": report["stage"],
                        "profile": profile,
                        "missing_sources": missing,
                        **row,
                    }
                )
                checkpoint()
                if row["status"] != "completed":
                    raise RuntimeError("product_turn_failed")

        multihop, correction, forget = plan["probes"]
        report["stage"] = multihop["id"]
        await pair(multihop, chat.id)
        correction_chat = await conversation.create_chat(owner, character, "correct")
        correction_turn = await conversation.accept(
            owner, correction_chat.id, "correct", correction["action"], model_mode="deepseek"
        )
        await conversation.cancel(owner, correction_turn.id, "end-correction-input")
        original = (await conversation.messages(owner, correction_chat.id))[0]
        sources[str(original.id)] = "probe-correction"
        report["stage"] = correction["id"] + "-immediate"
        await pair(correction, correction_chat.id)
        report["stage"] = correction["id"] + "-new-session"
        await pair(correction, None)

        report["stage"] = forget["id"]
        memory = Memory(database, digest_key=conversation.digest_key)
        fixture_source = next(UUID(k) for k, v in sources.items() if v == "probe-forget")
        receipts = []
        for source in (thread_source, fixture_source):
            receipts.append(
                await memory.forget_source(
                    owner, UUID(str(source)), uuid4().hex, expected_revision=1
                )
            )
        await framework.run_once(schedule_maintenance=False)
        report["forget_receipts"] = [
            (await memory.deletion(owner, receipt.id)).status for receipt in receipts
        ]
        report["timem_collection_removed"] = native.close()
        timem_closed = True
        if (
            report["forget_receipts"] != ["completed", "completed"]
            or not report["timem_collection_removed"]
        ):
            raise RuntimeError("forget_cleanup_incomplete")
        await database.close()
        report["reopened_after_forget"] = True
        await pair(forget, None)
        report["status"] = "completed"
    except Exception as error:
        report["status"] = "failed"
        report["error"] = evaluation_error_code(error)
        report["error_type"] = type(error).__name__
    finally:
        if getattr(native, "client", None) and not timem_closed:
            report["timem_collection_removed"] = native.close()
        await llm.close()
        for service in services.values():
            await service.close()
        if report["stage"] != "import":
            # Retire any still-ready evaluation space and drain its native
            # storage before dropping the application metadata.
            async with database.sessions.begin() as session:
                ids = list(await session.scalars(select(MemoryBuild.id)))
                await builds._retire(session, ids)
            for _ in ids:
                await framework.run_once(schedule_maintenance=False)
            async with database.sessions() as session:
                statuses = await session.scalars(select(MemorySpace.status))
                report["mem0_cleaned"] = all(status == "cleaned" for status in statuses)
        if report.get("mem0_cleaned", True):
            async with database.engine.begin() as connection:
                await connection.execute(text(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE'))
        await database.close()
        checkpoint()
    print(json.dumps({"status": report["status"], "stage": report["stage"]}))
    return 0 if report["status"] == "completed" else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", help="New slice-* artifact directory")
    parser.add_argument(
        "--budget", nargs=4, type=int, metavar=("TOTAL", "MEM0", "TIMEM", "PRODUCT")
    )
    args = parser.parse_args()
    sys.path.insert(0, str(TIMEM))
    sys.path.insert(0, str(ROOT / "artifacts/references/mem0"))
    sys.path.append(str(ROOT / "artifacts/framework-env/Lib/site-packages"))
    plan = json.loads(PLAN.read_bytes())
    if args.budget:
        total, mem0, timem, product = args.budget
        if min(args.budget) <= 0 or mem0 + timem + product > total:
            parser.error("Stage budgets must be positive and fit within the total")
        plan["budget"].update(
            maximum_calls=total,
            mem0_writer_max_calls=mem0,
            timem_writer_max_calls=timem,
            product_max_calls=product,
        )
    directory = PLAN.parent / args.directory
    if directory.parent.resolve() != PLAN.parent or not directory.name.startswith("slice-"):
        raise ValueError("Use a new slice-* artifact directory")
    directory.mkdir(exist_ok=False)
    save(
        directory / "run.json",
        {
            "plan_sha256": digest(PLAN.read_bytes()),
            "persona_sha256": digest(PERSONA.encode()),
            "budget": plan["budget"],
            "code_sha256": {
                str(file.relative_to(ROOT)): digest(file.read_bytes())
                for file in (
                    Path(__file__).resolve(),
                    ROOT / "src/hmph/deepseek.py",
                    ROOT / "src/hmph/companion.py",
                    ROOT / "src/hmph/timem_reference.py",
                    ROOT / "src/hmph/mem0_backend.py",
                    ROOT / "src/hmph/online_memory.py",
                    ROOT / "src/hmph/conversation.py",
                    ROOT / "src/hmph/worker.py",
                    ROOT / "src/hmph/shared_threads.py",
                    ROOT / "scripts/public_product.py",
                    TIMEM / "config/prompts.yaml",
                )
            },
        },
    )
    return asyncio.run(execute(directory, plan))


if __name__ == "__main__":
    raise SystemExit(main())
