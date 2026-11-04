"""Explicit, frozen C2 companion demo in a temporary database schema."""

import argparse
import asyncio
import json
from pathlib import Path
from uuid import uuid4

from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine

from hmph.config import Settings
from hmph.conversation import Conversation
from hmph.database import Database

ROOT = Path(__file__).resolve().parents[1]


async def shared_threads_demo(directory, execute):
    """New C2-only demo; old state smoke is never implicitly rerun."""
    import hashlib

    from hmph.deepseek import DeepSeekModel
    from hmph.eval_budget import RunBudget
    from hmph.product_eval import ProductTransport
    from hmph.shared_threads import SharedThreads
    from hmph.timem_reference import local_test_key
    from hmph.worker import Worker

    root = ROOT
    files = [*root.glob("src/hmph/**/*.py"), Path(__file__)]
    hashes = {str(f.relative_to(root)): hashlib.sha256(f.read_bytes()).hexdigest() for f in files}
    plan = {
        "suite": "shared-thread-product-demo-v1",
        "code_hashes": hashes,
        "model": "deepseek-flash",
        "topic": "每周一起聊聊练琴进展",
        "human_review": "not_run",
        "utterances": [
            "我回来了，今天想和你聊聊练琴。",
            "今天下班了，有点累。",
            "我回来了，继续聊聊练琴的事吧。",
            "晚安，今天先到这里。",
        ],
        "limits": {"max_calls": 8, "max_input_bytes": 128 * 1024, "per_case_calls": 8},
        "stopping": "first technical failure, STOP or budget exhaustion; no paid retry",
    }
    if not execute:
        directory.mkdir(parents=True, exist_ok=False)
        (directory / "plan.json").write_text(
            json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print("Prepared new C2 demo; real calls=0")
        return
    frozen = json.loads((directory / "plan.json").read_text(encoding="utf-8"))
    if frozen != plan or (directory / "report.json").exists():
        raise ValueError("Frozen demo changed or already started")
    settings = Settings()
    key = local_test_key(ROOT)
    budget = RunBudget(directory / "calls.sqlite", **plan["limits"])
    transports = {
        stage: ProductTransport(
            key,
            budget=budget.bind("p5-dev-001", "lexical", stage),
            stop_file=directory / "STOP",
        )
        for stage in ("generation", "guard")
    }
    models = {
        stage: DeepSeekModel(key, transport=transport) for stage, transport in transports.items()
    }
    schema = "live_c2_" + uuid4().hex
    admin = create_async_engine(settings.database_url.get_secret_value())
    db = Database(settings.database_url.get_secret_value(), schema=schema)
    report = {"status": "running", "rows": [], "human_review": "not_run"}
    async with admin.begin() as connection:
        await connection.execute(text(f'CREATE SCHEMA "{schema}"'))
    try:
        await db.migrate()
        conversation = Conversation(db, digest_key=b"synthetic-c2-only-request-digest")
        identity = await conversation.bootstrap()
        chat = await conversation.create_chat(identity.owner_id, identity.character_id, "first")
        controls = SharedThreads(conversation)
        thread = await controls.write(
            identity.owner_id, chat.id, "create", topic=plan["topic"], model_mode="deepseek"
        )
        chat = await conversation.create_chat(identity.owner_id, identity.character_id, "second")
        worker = Worker(
            conversation,
            models["generation"],
            model_mode="deepseek",
            model_name="deepseek-flash",
            reviewer=models["guard"].review,
        )
        from uuid import UUID

        for index, utterance in enumerate(plan["utterances"]):
            if index in (1, 2):
                thread = await controls.write(
                    identity.owner_id,
                    chat.id,
                    f"control-{index}",
                    thread_id=UUID(thread["id"]),
                    expected_revision=thread["revision"],
                    status="paused" if index == 1 else "open",
                    model_mode="deepseek",
                )
            if (directory / "STOP").exists():
                raise RuntimeError("Stopped")
            turn = await conversation.accept(
                identity.owner_id, chat.id, f"demo-{index}", utterance, model_mode="deepseek"
            )
            await worker.run_once()
            saved = await conversation.get_turn(identity.owner_id, turn.id)
            messages = await conversation.messages(identity.owner_id, chat.id)
            reply = next(
                (m.content for m in messages if m.turn_id == turn.id and m.role == "assistant"), ""
            )
            explanation = await conversation.companion_explanation(identity.owner_id, turn.id)
            report["rows"].append(
                {
                    "utterance": utterance,
                    "reply": reply,
                    "status": saved.status,
                    "error_code": saved.error_code,
                    "usage": saved.usage,
                    "plan": explanation["plan"],
                }
            )
            (directory / "report.json").write_text(
                json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            if saved.status != "completed":
                raise RuntimeError("Technical turn failure")
        report["status"] = "completed"
    except Exception as error:
        report.update(status="failed", error_type=type(error).__name__)
    finally:
        for transport in transports.values():
            await transport.shutdown()
        await db.close()
        async with admin.begin() as connection:
            await connection.execute(text(f'DROP SCHEMA "{schema}" CASCADE'))
        await admin.dispose()
        report["temporary_schema_removed"] = True
        report["budget"] = budget.snapshot()
        (directory / "report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    print(
        json.dumps(
            {
                "status": report["status"],
                "turns": len(report["rows"]),
                "calls": len(report["budget"]["calls"]),
            }
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--shared-threads", type=Path, required=True, help="new isolated C2 demo directory"
    )
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    asyncio.run(shared_threads_demo(args.shared_threads, args.run))
