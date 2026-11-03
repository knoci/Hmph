import asyncio

import httpx

TOKEN = "test-only-token-at-least-32-characters"
ORIGIN = "http://127.0.0.1:8017"


def browser_app(conversation, **extra):
    from hmph.api import create_app
    from hmph.config import Settings

    return create_app(
        Settings(
            api_token=TOKEN,
            request_digest_key="digest-key-at-least-32-characters",
            _env_file=None,
            **extra,
        ),
        conversation=conversation,
    )


async def test_browser_login_requires_origin_cookie_writes_require_csrf_and_logout_revokes(
    conversation,
):
    app = browser_app(conversation)
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url=ORIGIN
        ) as client:
            assert (await client.get("/")).status_code == 200
            assert (await client.post("/api/session", json={"token": TOKEN})).status_code == 403
            headers = {"Origin": ORIGIN}
            login = await client.post("/api/session", json={"token": TOKEN}, headers=headers)
            assert login.status_code == 200
            assert "httponly" in login.headers["set-cookie"].lower()
            assert "samesite=strict" in login.headers["set-cookie"].lower()
            assert TOKEN not in login.text
            csrf = login.json()["csrf_token"]
            assert (await client.get("/api/identity")).status_code == 200
            body = {"request_id": "browser-chat"}
            assert (
                await client.post("/api/conversations", json=body, headers=headers)
            ).status_code == 403
            headers["X-CSRF-Token"] = csrf
            assert (
                await client.post("/api/conversations", json=body, headers=headers)
            ).status_code == 201
            assert (
                await client.post(
                    "/api/session/logout", headers={**headers, "Origin": "https://bad.invalid"}
                )
            ).status_code == 403
            old_cookie = client.cookies.get("hmph_session")
            assert (await client.post("/api/session/logout", headers=headers)).status_code == 200
            client.cookies.set("hmph_session", old_cookie)
            assert (await client.get("/api/identity")).status_code == 401
            client.cookies.clear()
            assert (
                await client.post(
                    "/api/conversations",
                    json={"request_id": "script-chat"},
                    headers={"Authorization": f"Bearer {TOKEN}"},
                )
            ).status_code == 201


async def test_preference_source_is_forgettable_and_invalidates_old_turns(
    conversation,
):
    app = browser_app(conversation)
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url=ORIGIN,
            headers={"Authorization": f"Bearer {TOKEN}"},
        ) as client:
            chat = (await client.post("/api/conversations", json={"request_id": "chat"})).json()
            turn = (
                await client.post(
                    f"/api/conversations/{chat['id']}/turns",
                    json={"request_id": "old", "content": "你好"},
                )
            ).json()
            body = {
                "request_id": "pref",
                "conversation_id": chat["id"],
                "expected_revision": 1,
                "preferences": {
                    "nickname": "小木",
                    "reply_length": "short",
                    "playfulness": "low",
                    "support_mode": "listen",
                },
            }
            changed = await client.put("/api/preferences", json=body)
            assert changed.status_code == 200
            pref = changed.json()
            assert pref["revision"] == 2 and pref["preferences"]["nickname"] == "小木"
            assert (await client.get(f"/api/turns/{turn['id']}")).json()["status"] == "invalidated"
            assert (await client.put("/api/preferences", json=body)).json() == pref
            assert (
                await client.put("/api/preferences", json={**body, "request_id": "stale"})
            ).status_code == 409
            history = (await client.get(f"/api/conversations/{chat['id']}/messages")).json()
            assert any("小木" in row["content"] for row in history)
            active = (
                await client.post(
                    f"/api/conversations/{chat['id']}/turns",
                    json={"request_id": "new", "content": "今天有点累"},
                )
            ).json()
            lease = await conversation.claim("pref-test", 15)
            request = await conversation.generation_request(lease)
            assert request.preferences.nickname == "小木"
            forgotten = await client.post(
                f"/api/memories/{pref['source_memory_id']}/forget",
                json={"request_id": "forget-pref", "expected_revision": 1},
            )
            assert forgotten.status_code == 200
            current = (await client.get("/api/preferences")).json()
            assert current["preferences"]["nickname"] is None
            assert (await client.get(f"/api/turns/{active['id']}")).json()[
                "status"
            ] == "invalidated"
            assert (await client.put("/api/preferences", json=body)).json()["preferences"][
                "nickname"
            ] is None


async def test_browser_pages_are_bounded_and_active_turn_can_be_resumed(conversation):
    app = browser_app(conversation)
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url=ORIGIN,
            headers={"Authorization": f"Bearer {TOKEN}"},
        ) as client:
            chats = [
                (await client.post("/api/conversations", json={"request_id": f"c{i}"})).json()
                for i in range(3)
            ]
            first = (await client.get("/api/conversations/page?limit=2")).json()
            second = (
                await client.get(
                    "/api/conversations/page", params={"limit": 2, "cursor": first["next_cursor"]}
                )
            ).json()
            assert len(first["items"]) == 2 and len(second["items"]) == 1
            assert {c["id"] for c in first["items"] + second["items"]} == {c["id"] for c in chats}
            chat_id = chats[0]["id"]
            for i in range(3):
                turn = (
                    await client.post(
                        f"/api/conversations/{chat_id}/turns",
                        json={"request_id": f"t{i}", "content": f"消息{i}"},
                    )
                ).json()
                if i < 2:
                    await client.post(
                        f"/api/turns/{turn['id']}/cancel", json={"request_id": f"cancel{i}"}
                    )
            assert (await client.get("/api/active-turn")).json()["id"] == turn["id"]
            page = (await client.get(f"/api/conversations/{chat_id}/messages/page?limit=2")).json()
            older = (
                await client.get(
                    f"/api/conversations/{chat_id}/messages/page",
                    params={"cursor": page["next_cursor"], "limit": 2},
                )
            ).json()
            assert len(page["items"]) == 2 and len(older["items"]) == 1
            assert older["items"][0]["source_seq"] < page["items"][0]["source_seq"]
            assert (
                await client.get(f"/api/conversations/{chat_id}/messages/page?limit=101")
            ).status_code == 422


async def test_browser_session_expires_and_login_errors_do_not_echo_credentials(conversation):
    app = browser_app(conversation, browser_session_seconds=1)
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url=ORIGIN
        ) as client:
            login = await client.post(
                "/api/session", json={"token": TOKEN}, headers={"Origin": ORIGIN}
            )
            assert login.status_code == 200
            malformed = await client.post(
                "/api/session", json={"token": TOKEN * 100}, headers={"Origin": ORIGIN}
            )
            assert malformed.status_code == 422 and TOKEN not in malformed.text
            await asyncio.sleep(1.05)
            assert (await client.get("/api/identity")).status_code == 401
            for _ in range(9):
                assert (
                    await client.post(
                        "/api/session", json={"token": "wrong"}, headers={"Origin": ORIGIN}
                    )
                ).status_code == 401
            assert (
                await client.post("/api/session", json={"token": TOKEN}, headers={"Origin": ORIGIN})
            ).status_code == 429


async def test_memory_versions_and_sources_have_bounded_continuation(conversation):
    from hmph.memory import Memory
    from hmph.memory_types import MemoryProposal

    app = browser_app(conversation)
    async with app.router.lifespan_context(app):
        identity = app.state.identity
        memory = Memory(conversation.db, digest_key=conversation.digest_key)
        for i, value in enumerate(("北山", "北山", "南谷")):
            chat = await conversation.create_chat(
                identity.owner_id, identity.character_id, f"chat{i}"
            )
            turn = await conversation.accept(
                identity.owner_id, chat.id, f"turn{i}", f"我在{value}工作。"
            )
            await conversation.cancel(identity.owner_id, turn.id, f"cancel{i}")
            await memory.learn(
                await memory.claim("source-test"),
                [MemoryProposal(predicate="employer", value=value, quote=f"我在{value}工作")],
            )
        entry = (await memory.entries(identity.owner_id, identity.character_id))[0]
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url=ORIGIN,
            headers={"Authorization": f"Bearer {TOKEN}"},
        ) as client:
            current = (await client.get(f"/api/memories/{entry.id}/history/page?limit=1")).json()
            old = (
                await client.get(
                    f"/api/memories/{entry.id}/history/page",
                    params={"limit": 1, "cursor": current["next_cursor"]},
                )
            ).json()
            assert current["items"][0]["value"] == "南谷"
            assert old["items"][0]["value"] == "北山"
            first = (
                await client.get(f"/api/memories/{entry.id}/sources?revision=1&limit=1")
            ).json()
            second = (
                await client.get(
                    f"/api/memories/{entry.id}/sources",
                    params={"revision": 1, "limit": 1, "cursor": first["next_cursor"]},
                )
            ).json()
            assert first["items"][0]["message_id"] != second["items"][0]["message_id"]
            assert second["next_cursor"] is None
