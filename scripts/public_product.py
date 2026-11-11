"""P4-e paired recent-history and product-memory read-path comparison.

Historical documents are imported into an isolated schema, with their original
observation clocks. No benchmark answer is used in import, learning or generation.
"""

import argparse
import asyncio
import hashlib
import json
import sys
import time
from datetime import timedelta
from pathlib import Path
from uuid import uuid4

from pydantic import RootModel
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from product_run import CapturingConversation, Services, evaluation_error_code, save  # noqa: E402
from public_memory import pack  # noqa: E402
from timem_public import JUDGE  # noqa: E402

from hmph.contracts import DomainError  # noqa: E402
from hmph.database import Database  # noqa: E402
from hmph.dialogue import Dialogue, queue_dialogue  # noqa: E402
from hmph.dialogue_types import ShortTermContext  # noqa: E402
from hmph.dialogue_worker import StateWorker  # noqa: E402
from hmph.eval_budget import RunBudget  # noqa: E402
from hmph.memory import Memory  # noqa: E402
from hmph.models import Chat, MemoryTask, Message, Pair, Turn  # noqa: E402
from hmph.preference_types import ExpressionPreferences  # noqa: E402
from hmph.product_eval import RunControl  # noqa: E402
from hmph.selection_eval import source_text  # noqa: E402
from hmph.timem_reference import local_test_key  # noqa: E402
from hmph.worker import Worker  # noqa: E402


def paired_memory_requests(request):
    if request.messages[-1].role != "user":
        raise ValueError("Paired memory requests require a current user message")
    user_positions = [i for i, message in enumerate(request.messages) if message.role == "user"]
    start = user_positions[max(0, len(user_positions) - 5)]
    full = request.model_copy(
        update={
            "messages": request.messages[start:],
            "history_origins": request.history_origins[start:],
        }
    )
    recent_only = full.model_copy(
        update={
            "evidence": (),
            "raw_evidence": (),
            "short_term": ShortTermContext(),
            "workspace": None,
            "preferences": ExpressionPreferences(),
            "companion": None,
        }
    )
    return full, recent_only


async def import_history(db, chat_id, case, *, format_source=source_text):
    """Fixture import only: canonical sources/tasks, never generated facts or state."""
    async with db.sessions.begin() as session:
        chat = await session.get(Chat, chat_id)
        pair = await session.get(Pair, chat.pair_id)
        turns, messages, tasks = [], [], []
        for sequence, source in enumerate(case.history, 1):
            turn = Turn(
                id=uuid4(),
                conversation_id=chat.id,
                pair_id=pair.id,
                status="completed",
                source_seq=sequence,
                policy_revision=pair.policy_revision,
                memory_epoch=pair.memory_epoch,
                source_timezone="UTC",
                created_at=source.observed_at,
                deadline=source.observed_at + timedelta(seconds=45),
                model_mode="deepseek",
                usage={"benchmark_document_import": True, "generation_calls": 0},
            )
            message = Message(
                id=uuid4(),
                conversation_id=chat.id,
                turn_id=turn.id,
                source_seq=sequence,
                role="user",
                content=format_source(source),
                revision=1,
            )
            turns.append(turn)
            messages.append(message)
            tasks.append(
                MemoryTask(
                    pair_id=pair.id,
                    source_message_id=message.id,
                    source_revision=1,
                    source_order=sequence,
                    memory_epoch=pair.memory_epoch,
                    model_mode="deepseek",
                )
            )
        session.add_all(turns)
        await session.flush()
        session.add_all(messages)
        await session.flush()
        session.add_all(tasks)
        chat.next_source_seq = len(messages) + 1
        pair.next_memory_seq = len(messages) + 1
        # The short-term layer starts at a declared recent suffix, without paid history backfill.
        await queue_dialogue(session, turns[max(0, len(turns) - 4)], len(messages))
        return {str(m.id): s.source_id for m, s in zip(messages, case.history, strict=True)}


async def main(directory, execute):
    reference = ROOT / "artifacts/evals/p4e-public-native-dev-v3/plan.json"
    previous = json.loads(reference.read_text(encoding="utf-8"))
    files = [
        Path(__file__),
        ROOT / "scripts/product_run.py",
        ROOT / "scripts/public_memory.py",
        *ROOT.glob("src/hmph/**/*.py"),
    ]
    config = {
        "package": "P4-e",
        "purpose": "paired product memory baseline",
        "reference_sha256": hashlib.sha256(reference.read_bytes()).hexdigest(),
        "model": "deepseek-flash",
        "backend": "lexical",
        "temperature": 0,
        "variants": ["recent_only", "full"],
        "recent_turns": 4,
        "metric": "semantic answer correctness; one history/question per dataset",
        "interpretation": "development only; inconclusive cannot pass design 10.4",
        "limits": {"max_calls": 1024, "max_input_bytes": 32 * 1024 * 1024, "per_case_calls": 512},
        "database": "postgresql+asyncpg://hmph:hmph_local_only@127.0.0.1:15439/hmph",
        "fixture": "native-matched role/time documents; original clocks; recent-four short state",
        "pairing": "same recent turns and query; memory channels present only in full",
        "latency_target_seconds": 45,
        "human_review": "not_run",
        "hashes": {
            str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files
        },
        "stop": "provider/technical failure or budget cap; rejected facts retained; no retries",
    }
    if not execute:
        directory.mkdir(parents=True, exist_ok=False)
        save(directory / "config.json", config)
        print("P4-e product configuration frozen; real calls=0")
        return
    if (directory / "report.json").exists() or config != json.loads(
        (directory / "config.json").read_text(encoding="utf-8")
    ):
        raise ValueError("Frozen configuration changed or already started")
    budget = RunBudget(directory / "calls.sqlite", **config["limits"])
    key = local_test_key(ROOT)
    report = {"status": "running", "rows": [], "paper_decision": "pending"}

    def checkpoint():
        save(directory / "report.json", {**report, "budget": budget.snapshot()})

    checkpoint()
    try:
        for index, record in enumerate(previous["records"], 1):
            case, mapping = pack(record)
            services = Services(
                key, budget, f"case-{index:03}", "lexical", control=RunControl(directory / "STOP")
            )
            schema = "p4e_product_" + uuid4().hex
            db = Database(config["database"], schema=schema)
            admin = create_async_engine(config["database"])
            row = {
                "dataset": record["dataset"],
                "status": "running",
                "answers": [],
                "rejected_sources": [],
                "schema_removed": False,
            }
            report["rows"].append(row)
            lease = None
            heartbeat = None
            try:
                async with admin.begin() as connection:
                    await connection.execute(text(f'CREATE SCHEMA "{schema}"'))
                await db.migrate()
                conversation = CapturingConversation(
                    db, digest_key=b"p4e-public-product", turn_timeout=180, source_timezone="UTC"
                )
                identity = await conversation.bootstrap()
                chat = await conversation.create_chat(
                    identity.owner_id, identity.character_id, uuid4().hex
                )
                source_map = await import_history(db, chat.id, case)
                row["source_mapping"] = source_map
                row["episode_sources"] = mapping
                memory = Memory(db, digest_key=conversation.digest_key)
                began = time.monotonic()
                for _ in case.history:
                    services.check_stop()
                    task = await memory.claim("p4e-extract", 60, model_mode="deepseek")
                    if task is None:
                        continue
                    proposals, usage = await services.model("extract").extract(task.content)
                    try:
                        learned = await memory.learn(
                            task, proposals, usage=usage.model_dump() if usage else None
                        )
                        if learned.status != "committed":
                            raise RuntimeError("learning_lease_lost")
                    except DomainError:
                        await memory.fail(
                            task,
                            code="unsupported_memory",
                            usage=usage.model_dump() if usage else None,
                        )
                        row["rejected_sources"].append(str(task.id))
                    services.check_budget()
                    row["extracted_sources"] = row.get("extracted_sources", 0) + 1
                    checkpoint()
                row["long_write_seconds"] = time.monotonic() - began
                state = Dialogue(db)
                updater = StateWorker(
                    state, services.model("state").update_state, model_mode="deepseek"
                )
                for _ in range(8):
                    current = await state.read(identity.owner_id, chat.id)
                    if current.task_status != "pending":
                        break
                    await updater.run_once()
                    services.check_budget()
                row["short_state"] = (await state.read(identity.owner_id, chat.id)).model_dump(
                    mode="json"
                )
                if row["short_state"]["task_status"] != "idle":
                    raise RuntimeError("short_state_not_ready")
                accepted = await conversation.accept(
                    identity.owner_id,
                    chat.id,
                    uuid4().hex,
                    case.query.question,
                    model_mode="deepseek",
                )
                async with db.sessions.begin() as session:
                    query_turn = await session.get(Turn, accepted.id)
                    query_turn.created_at = case.query.timestamp
                lease = await conversation.claim("p4e-product", 30, model_mode="deepseek")
                if lease is None:
                    raise RuntimeError("query_not_claimed")
                worker = Worker(
                    conversation,
                    services.model("generation"),
                    model_mode="deepseek",
                    lease_seconds=30,
                )
                heartbeat = asyncio.create_task(worker._heartbeat(lease))
                request, recent_only = paired_memory_requests(
                    await conversation.generation_request(lease)
                )
                row["actual_input"] = request.model_dump(mode="json")
                for variant in config["variants"]:
                    services.check_stop()
                    chosen = recent_only if variant == "recent_only" else request
                    began = time.monotonic()
                    body = ""
                    async for delta in services.model("generation").stream(chosen):
                        body += delta.text
                        if len(body) > 2000:
                            raise RuntimeError("output_limit")
                    answer_row = {
                        "variant": variant,
                        "reply": body,
                        "actual_input": chosen.model_dump(mode="json"),
                        "read_seconds": time.monotonic() - began,
                    }
                    row["answers"].append(answer_row)
                    checkpoint()
                    grade, _ = await services.model("answer_judge")._evaluation_json(
                        JUDGE,
                        RootModel(
                            {
                                "question": record["question"],
                                "reference": record["answer"],
                                "hypothesis": body,
                            }
                        ),
                        max_tokens=128,
                    )
                    verdict = json.loads(grade)
                    if type(verdict.get("correct")) is not bool:
                        raise ValueError("Invalid correctness grade")
                    answer_row["correct"] = verdict["correct"]
                    services.check_budget()
                    checkpoint()
                await conversation.cancel(identity.owner_id, accepted.id, uuid4().hex)
                row["status"] = "completed"
            finally:
                if heartbeat:
                    heartbeat.cancel()
                    await asyncio.gather(heartbeat, return_exceptions=True)
                if lease:
                    await conversation.recover_expired()
                await services.close()
                await db.close()
                async with admin.begin() as connection:
                    await connection.execute(text(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE'))
                row["schema_removed"] = True
                await admin.dispose()
                checkpoint()
        report["status"] = "completed"
        report["paper_decision"] = (
            "inconclusive: one history per dataset cannot establish clustered -5pp bound"
        )
    except BaseException as error:
        report["status"] = "failed"
        report["error_code"] = evaluation_error_code(error)
        report["paper_decision"] = "inconclusive: missing product observations"
    finally:
        checkpoint()
    print(json.dumps({"status": report["status"], "paper_decision": report["paper_decision"]}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", required=True, type=Path)
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    asyncio.run(main(args.directory, args.run))
