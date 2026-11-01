"""Opt-in real DeepSeek smoke, synthetic data, temporary PostgreSQL schema.

Run from the repository root. This is a wiring check, not a memory benchmark.
It makes two extraction calls plus three generation/review pairs on success.
Credentials are loaded only through Settings and never written to the report.
"""

import argparse
import asyncio
import json
import time
from pathlib import Path
from uuid import uuid4

from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine

from hmph.config import Settings
from hmph.conversation import Conversation
from hmph.database import Database
from hmph.memory import Memory
from hmph.runtime import build_memory_worker, build_worker


async def run(output: Path):
    settings = Settings()
    if settings.model_mode != "deepseek":
        raise RuntimeError("This opt-in smoke requires local DeepSeek mode")
    schema = "live_memory_" + uuid4().hex
    admin = create_async_engine(settings.database_url.get_secret_value())
    db = Database(settings.database_url.get_secret_value(), schema=schema)
    report = {
        "suite": "synthetic_memory_wiring_v1",
        "model": settings.deepseek_model,
        "benchmark": False,
        "frontier_claim": False,
        "cases": [],
        "extraction_usage": [],
    }
    async with admin.begin() as connection:
        await connection.execute(text(f'CREATE SCHEMA "{schema}"'))
    try:
        await db.migrate()
        conversation = Conversation(
            db,
            digest_key=settings.request_digest_key.get_secret_value().encode(),
            source_timezone=settings.source_timezone,
        )
        identity = await conversation.bootstrap()
        memory = Memory(
            db, digest_key=conversation.digest_key, source_timezone=settings.source_timezone
        )
        extractor = build_memory_worker(conversation, settings)
        generator = build_worker(conversation, settings)

        async def ingest(key, content):
            chat = await conversation.create_chat(identity.owner_id, identity.character_id, key)
            turn = await conversation.accept(
                identity.owner_id, chat.id, key, content, model_mode="deepseek"
            )
            await conversation.cancel(identity.owner_id, turn.id, key)
            source = (await conversation.messages(identity.owner_id, chat.id))[0]
            await extractor.run_once()
            task = await memory.task_status(identity.owner_id, source.id)
            if task.status != "done" or task.error_code:
                raise RuntimeError("Synthetic extraction did not commit")
            report["extraction_usage"].append(task.usage)

        async def ask(key, question, required=(), forbidden=()):
            chat = await conversation.create_chat(identity.owner_id, identity.character_id, key)
            started = time.monotonic()
            turn = await conversation.accept(
                identity.owner_id, chat.id, key, question, model_mode="deepseek"
            )
            await generator.run_once()
            saved = await conversation.get_turn(identity.owner_id, turn.id)
            messages = await conversation.messages(identity.owner_id, chat.id)
            answer = messages[-1].content if messages[-1].role == "assistant" else ""
            passed = (
                saved.status == "completed"
                and all(v in answer for v in required)
                and all(v not in answer for v in forbidden)
            )
            report["cases"].append(
                {
                    "id": key,
                    "passed": passed,
                    "answer": answer,
                    "seconds": round(time.monotonic() - started, 2),
                    "usage": saved.usage,
                    "error_code": saved.error_code,
                }
            )
            if not passed:
                raise RuntimeError("Synthetic response check failed")

        await ingest("fact", "我叫雾舟。我现在在赤松研究社工作。")
        await ingest("update", "现在我换工作了，在白鹭实验室工作。")
        name = next(
            (
                v
                for v in await memory.entries(identity.owner_id, identity.character_id)
                if v.predicate == "name"
            ),
            None,
        )
        if name is None:
            raise RuntimeError("Synthetic name was not extracted")
        await ask(
            "current",
            "我叫什么？现在在哪家公司工作？不知道就直说。",
            required=("雾舟", "白鹭实验室"),
            forbidden=("赤松研究社",),
        )
        await ask("past", "我以前在哪家公司工作？不知道就直说。", required=("赤松研究社",))
        preview = await memory.preview_forget(identity.owner_id, name.id)
        receipt = await memory.forget(
            identity.owner_id, name.id, "forget", expected_revision=name.revision
        )
        report["forget"] = {
            "preview": preview.model_dump(mode="json"),
            "receipt": receipt.model_dump(mode="json"),
        }
        if await memory.recall(identity.owner_id, identity.character_id, "名字公司"):
            raise RuntimeError("Synthetic memory remained after deletion")
        await ask("after_forget", "我叫什么？不知道就直说。", forbidden=("雾舟",))
        report["status"] = "passed"
    except Exception:
        report["status"] = "failed"
        # Do not serialize exceptions: SDK/transport errors may contain request data.
        raise RuntimeError("Memory smoke failed; inspect sanitized report") from None
    finally:
        await db.close()
        async with admin.begin() as connection:
            await connection.execute(text(f'DROP SCHEMA "{schema}" CASCADE'))
        await admin.dispose()
        report["temporary_schema_removed"] = True
        await asyncio.to_thread(output.parent.mkdir, parents=True, exist_ok=True)
        await asyncio.to_thread(
            output.write_text, json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    print("Synthetic memory smoke passed. This is not a benchmark score.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("artifacts/live/memory-smoke.json"))
    asyncio.run(run(parser.parse_args().output))
