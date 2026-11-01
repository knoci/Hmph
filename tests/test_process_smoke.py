import asyncio
import os
import socket
import subprocess
import sys

import httpx


async def test_real_api_and_worker_process_survive_a_worker_crash(conversation):
    token = "process-test-only-token-at-least-32-characters"
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    environment = dict(
        os.environ,
        HMPH_API_TOKEN=token,
        HMPH_REQUEST_DIGEST_KEY="process-test-only-digest-key-at-least-32-chars",
        HMPH_DATABASE_SCHEMA=conversation.db.schema,
        HMPH_DATABASE_URL=conversation.db.url,
        HMPH_FAKE_DELAY_SECONDS="0.3",
        HMPH_LEASE_SECONDS="0.3",
        PYTHONUTF8="1",
    )

    def launch(*arguments):
        return subprocess.Popen(
            [sys.executable, "-m", "hmph.cli", *arguments],
            env=environment,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0,
        )

    children = [launch("serve", "--port", str(port))]
    try:
        async with httpx.AsyncClient(
            base_url=f"http://127.0.0.1:{port}",
            headers={"Authorization": f"Bearer {token}"},
            trust_env=False,
        ) as client:
            async with asyncio.timeout(15):
                for _ in range(150):
                    try:
                        if (await client.get("/ready")).status_code == 200:
                            break
                    except httpx.TransportError:
                        pass
                    await asyncio.sleep(0.05)
                else:
                    raise AssertionError("API process did not become ready")
                chat = (await client.post("/api/conversations", json={"request_id": "chat"})).json()
                turn = (
                    await client.post(
                        f"/api/conversations/{chat['id']}/turns",
                        json={
                            "request_id": "turn",
                            "content": "进程崩溃演练。",
                        },
                    )
                ).json()
                crashed = launch("worker")
                children.append(crashed)
                for _ in range(200):
                    messages = (
                        await client.get(
                            f"/api/conversations/{chat['id']}/messages",
                        )
                    ).json()
                    if len(messages) == 2:
                        break
                    await asyncio.sleep(0.01)
                else:
                    raise AssertionError("Worker did not publish its first segment")
                crashed.kill()
                crashed.wait(timeout=5)
                children.append(launch("worker"))
                for _ in range(200):
                    saved = (await client.get(f"/api/turns/{turn['id']}")).json()
                    if saved["status"] == "failed":
                        break
                    await asyncio.sleep(0.02)
                else:
                    raise AssertionError("Replacement worker did not recover the abandoned turn")
                replay = await client.get(f"/api/turns/{turn['id']}/events")
                assert saved["error_code"] == "worker_lost"
                assert replay.text.count("event: text.delta") == 1
                assert "event: turn.failed" in replay.text
    finally:
        for process in reversed(children):
            if process.poll() is None:
                process.terminate()
                process.wait(timeout=5)
