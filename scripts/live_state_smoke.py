"""Opt-in synthetic DeepSeek state/delta smoke in a temporary database schema."""

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
from hmph.dialogue import Dialogue
from hmph.runtime import build_state_worker, build_worker


async def run(output):
    settings = Settings()
    if settings.model_mode != "deepseek":
        raise RuntimeError("Local DeepSeek mode required")
    schema = "live_state_" + uuid4().hex
    admin = create_async_engine(settings.database_url.get_secret_value())
    database = Database(settings.database_url.get_secret_value(), schema=schema)
    report = {
        "suite": "synthetic-state-delta-v1",
        "model": settings.deepseek_model,
        "benchmark": False,
        "frontier_claim": False,
        "status": "failed",
    }
    async with admin.begin() as connection:
        await connection.execute(text(f'CREATE SCHEMA "{schema}"'))
    try:
        await database.migrate()
        conversation = Conversation(
            database,
            digest_key=settings.request_digest_key.get_secret_value().encode(),
            source_timezone=settings.source_timezone,
        )
        identity = await conversation.bootstrap()
        dialogue = Dialogue(database)
        worker = build_state_worker(conversation, settings)
        chat = await conversation.create_chat(identity.owner_id, identity.character_id, "chat")
        first = await conversation.accept(
            identity.owner_id,
            chat.id,
            "plan",
            "周五我要去星河公司面试，先听我吐槽，不用给建议。",
            model_mode="deepseek",
        )
        await conversation.cancel(identity.owner_id, first.id, "cancel")
        await worker.run_once()
        before = await dialogue.read(identity.owner_id, chat.id)
        report["before"] = before.model_dump(mode="json")
        if before.task_status != "idle" or before.data.goal is None or not before.data.open_items:
            raise RuntimeError("Initial synthetic state did not commit")

        turn = await conversation.accept(
            identity.owner_id,
            chat.id,
            "correction",
            "取消周五星河公司的面试，改成周日去远山公司面试。现在我安排在哪天哪家公司？",
            model_mode="deepseek",
        )
        # Deliberately let state lag: the raw correction must win over the old goal.
        await build_worker(conversation, settings).run_once()
        saved = await conversation.get_turn(identity.owner_id, turn.id)
        messages = await conversation.messages(identity.owner_id, chat.id)
        answer = messages[-1].content if messages[-1].role == "assistant" else ""
        report["answer"], report["generation_usage"] = answer, saved.usage
        if saved.status != "completed" or "周日" not in answer or "远山" not in answer:
            raise RuntimeError("Raw correction was not used")
        await worker.run_once()
        after = await dialogue.read(identity.owner_id, chat.id)
        report["after"] = after.model_dump(mode="json")
        if (
            after.task_status != "idle"
            or after.processed_through_seq != 4
            or not any(v.status == "dropped" for v in after.data.open_items)
        ):
            raise RuntimeError("Cancelled arrangement did not close")
        report["status"] = "passed"
    except Exception:
        raise RuntimeError("State smoke failed; inspect sanitized report") from None
    finally:
        await database.close()
        async with admin.begin() as connection:
            await connection.execute(text(f'DROP SCHEMA "{schema}" CASCADE'))
        await admin.dispose()
        report["temporary_schema_removed"] = True
        await asyncio.to_thread(output.parent.mkdir, parents=True, exist_ok=True)
        await asyncio.to_thread(
            output.write_text, json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    print("Synthetic state/delta smoke passed; this is not a benchmark score.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("artifacts/live/state-smoke.json"))
    asyncio.run(run(parser.parse_args().output))
