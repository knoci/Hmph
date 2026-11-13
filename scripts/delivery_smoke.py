"""CLI/HTTP delivery check with disposable data and a fake model."""

import argparse
import asyncio
import hashlib
import json
import os
import socket
import subprocess
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path
from uuid import uuid4, uuid5

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

import httpx  # noqa: E402
from alembic import command  # noqa: E402
from dotenv import dotenv_values  # noqa: E402
from sqlalchemy import text  # noqa: E402

from hmph.config import Settings  # noqa: E402
from hmph.conversation import DEFAULT_CHARACTER, DEFAULT_OWNER  # noqa: E402
from hmph.database import Database  # noqa: E402
from hmph.models import Character, Chat, Message, Owner, Turn  # noqa: E402
from scripts.product_run import save  # noqa: E402


async def main(directory):
    directory = directory.resolve()
    if (
        not directory.is_relative_to(ROOT / "artifacts/delivery")
        or directory == ROOT / "artifacts/delivery"
        or directory.exists()
    ):
        raise ValueError("Use a new artifacts/delivery directory")
    directory.mkdir(parents=True)
    schema = "delivery_" + uuid4().hex
    url = Settings().database_url.get_secret_value()
    db = Database(url, schema=schema)
    report = {
        "status": "running",
        "schema": schema,
        "scope": "local fake-model CLI/HTTP; no human quality rating",
        "real_model_calls": 0,
        "checks": {},
    }
    processes = []
    created = False
    env = dict(os.environ)
    env.update(
        HMPH_DATABASE_URL=url,
        HMPH_DATABASE_SCHEMA=schema,
        HMPH_MODEL_MODE="fake",
        HMPH_DEEPSEEK_API_KEY="",
        HMPH_GRAPHITI_READER_PYTHON="",
    )
    # New local credentials never enter arguments, captured output, or reports.
    flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0

    def checkpoint(stage):
        report["stage"] = stage
        save(directory / "report.json", report)

    async def cli(*args):
        p = await asyncio.create_subprocess_exec(
            sys.executable,
            "-c",
            "from hmph.cli import main; main()",
            *args,
            cwd=directory,
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=flags,
        )
        processes.append(p)
        return await asyncio.wait_for(p.wait(), 30)

    async def start(*args):
        p = await asyncio.create_subprocess_exec(
            sys.executable,
            "-c",
            "from hmph.cli import main; main()",
            *args,
            cwd=directory,
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=flags,
        )
        processes.append(p)
        return p

    async def stop(p):
        if p.returncode is None:
            p.terminate()
        await asyncio.wait_for(p.wait(), 10)

    async def ready(client, status):
        end = asyncio.get_running_loop().time() + 15
        while asyncio.get_running_loop().time() < end:
            try:
                r = await client.get("/ready")
                if r.status_code == status:
                    return r
            except httpx.TransportError:
                pass
            await asyncio.sleep(0.1)
        raise RuntimeError("delivery_ready_timeout")

    async def complete(client, turn_id):
        end = asyncio.get_running_loop().time() + 15
        while asyncio.get_running_loop().time() < end:
            r = (await client.get(f"/api/turns/{turn_id}")).json()
            if r["status"] in {"completed", "failed", "invalidated", "cancelled"}:
                assert r["status"] == "completed"
                return
            await asyncio.sleep(0.1)
        raise RuntimeError("delivery_turn_timeout")

    try:
        checkpoint("fresh_init")
        assert await cli("init") == 0
        config = directory / ".env"
        initial = hashlib.sha256(config.read_bytes()).hexdigest()
        assert await cli("init") != 0 and hashlib.sha256(config.read_bytes()).hexdigest() == initial
        credentials = dotenv_values(config)
        env["HMPH_API_TOKEN"] = credentials["HMPH_API_TOKEN"]
        env["HMPH_REQUEST_DIGEST_KEY"] = credentials["HMPH_REQUEST_DIGEST_KEY"]
        report["checks"]["init_does_not_rotate_existing_config"] = True
        async with db.engine.begin() as c:
            await c.execute(text(f'CREATE SCHEMA "{schema}"'))
        created = True
        checkpoint("old_schema_and_existing_data")

        def old_migration(c):
            config = Database._migration_config()
            config.attributes.update(connection=c, schema=schema)
            command.upgrade(config, "0011")

        async with db.engine.begin() as c:
            await c.run_sync(old_migration)
        pair_id = uuid5(DEFAULT_OWNER, str(DEFAULT_CHARACTER))
        chat_id = uuid4()
        old_turn = uuid4()
        old_message = uuid4()
        async with db.sessions.begin() as s:
            await s.execute(
                Owner.__table__.insert().values(id=DEFAULT_OWNER, name="Delivery fixture owner")
            )
            await s.execute(
                Character.__table__.insert().values(
                    id=DEFAULT_CHARACTER, slug="hmph", persona_version=1
                )
            )
            await s.execute(
                text(
                    "INSERT INTO pairs "
                    "(id, owner_id, character_id, preferences, policy_revision, memory_epoch) "
                    "VALUES (:id, :owner_id, :character_id, '{}'::jsonb, 1, 1)"
                ),
                {"id": pair_id, "owner_id": DEFAULT_OWNER, "character_id": DEFAULT_CHARACTER},
            )
            await s.execute(
                Chat.__table__.insert().values(id=chat_id, pair_id=pair_id, next_source_seq=2)
            )
            statement = Turn.__table__.insert().values(
                id=old_turn,
                conversation_id=chat_id,
                pair_id=pair_id,
                status="cancelled",
                source_seq=1,
                policy_revision=1,
                memory_epoch=1,
                source_timezone="UTC",
                deadline=datetime.now(UTC) + timedelta(seconds=45),
            )
            assert "companion_plan" not in str(statement.compile())
            await s.execute(statement)
            await s.execute(
                Message.__table__.insert().values(
                    id=old_message,
                    conversation_id=chat_id,
                    turn_id=old_turn,
                    source_seq=1,
                    role="user",
                    content="upgrade-preserved-message",
                )
            )
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
        origin = f"http://127.0.0.1:{port}"
        server = await start("serve", "--port", str(port))
        async with httpx.AsyncClient(base_url=origin, timeout=8, trust_env=False) as client:
            assert await asyncio.wait_for(server.wait(), 15) != 0 and not await db.is_current()
            await stop(server)
            checkpoint("upgrade_cli")
            assert await cli("migrate") == 0 and await db.is_current()
            async with db.sessions() as s:
                assert (
                    await s.scalar(
                        text("SELECT content FROM messages WHERE id=:id"), {"id": old_message}
                    )
                    == "upgrade-preserved-message"
                )
            report["checks"]["upgrade_from_0011_preserves_existing_message"] = True
            report["checks"]["stale_schema_startup_rejected"] = True
            server = await start("serve", "--port", str(port))
            await ready(client, 200)
            assert (await client.get("/")).status_code == 200
            assert (await client.get("/static/app.js")).status_code == 200
            assert (await client.get("/api/identity")).status_code == 401
            client.headers["Origin"] = origin

            async def login():
                r = await client.post("/api/session", json={"token": credentials["HMPH_API_TOKEN"]})
                assert r.status_code == 200 and "httponly" in r.headers["set-cookie"].lower()
                client.headers["X-CSRF-Token"] = r.json()["csrf_token"]

            await login()
            worker = await start("worker")
            checkpoint("continuous_chat")
            chat = (
                await client.post("/api/conversations", json={"request_id": "delivery-chat"})
            ).json()
            body = {"request_id": "first", "content": "你好"}
            t = (await client.post(f"/api/conversations/{chat['id']}/turns", json=body)).json()
            await complete(client, t["id"])
            again = (await client.post(f"/api/conversations/{chat['id']}/turns", json=body)).json()
            assert again["id"] == t["id"]
            events = await client.get(f"/api/turns/{t['id']}/events")
            assert "text.delta" in events.text and "turn.completed" in events.text
            messages = (await client.get(f"/api/conversations/{chat['id']}/messages")).json()
            assert len([m for m in messages if m["role"] == "assistant"]) == 1
            report["checks"]["chat_sse_and_idempotent_replay"] = True
            checkpoint("restart_and_accepted_job_recovery")
            await stop(worker)
            t2 = (
                await client.post(
                    f"/api/conversations/{chat['id']}/turns",
                    json={"request_id": "second", "content": "继续聊聊"},
                )
            ).json()
            assert t2["status"] == "accepted"
            await stop(server)
            server = await start("serve", "--port", str(port))
            await ready(client, 200)
            assert (await client.get("/api/identity")).status_code == 401
            await login()
            worker = await start("worker")
            await complete(client, t2["id"])
            messages = (await client.get(f"/api/conversations/{chat['id']}/messages")).json()
            assert len([m for m in messages if m["role"] == "assistant"]) == 2
            report["checks"]["restart_recovers_accepted_job_without_duplicate_reply"] = True
            report["checks"]["restart_revokes_browser_session"] = True
            checkpoint("preferences_and_forget")
            pref = (
                await client.put(
                    "/api/preferences",
                    json={
                        "request_id": "pref",
                        "conversation_id": chat["id"],
                        "expected_revision": 1,
                        "preferences": {"nickname": "小杉"},
                    },
                )
            ).json()
            assert pref["preferences"]["nickname"] == "小杉"
            entries = (await client.get("/api/memories")).json()
            assert entries
            receipt = await client.post(
                f"/api/memories/{pref['source_memory_id']}/forget",
                json={"request_id": "forget", "expected_revision": 1},
            )
            assert receipt.status_code == 200 and receipt.json()["status"] == "completed"
            assert (await client.get("/api/preferences")).json()["preferences"]["nickname"] is None
            report["checks"]["manage_and_forget_preference_source"] = True
            # Original-message deletion must work without an extracted entry.
            messages = (await client.get(f"/api/conversations/{chat['id']}/messages")).json()
            source = next(m for m in messages if m["role"] == "user" and not m["deleted"])
            base = f"/api/messages/{source['id']}"
            preview = await client.get(base + "/forget-preview")
            assert preview.status_code == 200
            response = await client.post(
                base + "/forget",
                json={
                    "request_id": "forget-original",
                    "expected_revision": preview.json()["revision"],
                },
            )
            assert response.status_code == 200
            receipt = response.json()
            assert receipt["status"] == "completed"
            assert receipt["source_message_id"] == source["id"] and receipt["memory_id"] is None
            report["checks"]["forget_original_without_extracted_entry"] = True
        report["status"] = "completed"
    except Exception as error:
        report["status"] = "failed"
        report["error_type"] = type(error).__name__
    finally:
        stopped = True
        for p in reversed(processes):
            try:
                await stop(p)
            except Exception:
                stopped = False
        await db.close()
        report["owned_processes_stopped"] = stopped
        if created and stopped:
            admin = Database(url)
            try:
                async with admin.engine.begin() as c:
                    await c.execute(text("SET LOCAL lock_timeout = '5s'"))
                    await c.execute(text(f'DROP SCHEMA "{schema}" CASCADE'))
                report["schema_removed"] = True
            finally:
                await admin.close()
        save(directory / "report.json", report)
    print(
        json.dumps(
            {
                k: report.get(k)
                for k in [
                    "status",
                    "stage",
                    "checks",
                    "error_type",
                    "owned_processes_stopped",
                    "schema_removed",
                    "real_model_calls",
                ]
            },
            ensure_ascii=True,
        )
    )
    if report["status"] != "completed":
        raise SystemExit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    asyncio.run(main(parser.parse_args().directory))
