import httpx
import pytest


async def test_unextracted_message_forgetting_has_authenticated_preview_and_receipt(conversation):
    from hmph.api import create_app
    from hmph.config import Settings

    identity = await conversation.bootstrap()
    chat = await conversation.create_chat(identity.owner_id, identity.character_id, "raw")
    turn = await conversation.accept(identity.owner_id, chat.id, "fact", "我有一台星灯收音机。")
    await conversation.cancel(identity.owner_id, turn.id, "cancel")
    source = (await conversation.messages(identity.owner_id, chat.id))[0]
    token = "test-only-token-at-least-32-characters"
    app = create_app(
        Settings(
            api_token=token,
            request_digest_key="test-only-digest-key-at-least-32-characters",
            _env_file=None,
        ),
        conversation=conversation,
    )
    path = f"/api/messages/{source.id}"
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            assert (await client.get(path + "/forget-preview")).status_code == 401
            client.headers["Authorization"] = "Bearer " + token
            preview = await client.get(path + "/forget-preview")
            assert preview.status_code == 200 and preview.json()["memories"] == 0
            bad = await client.post(
                path + "/forget", json={"request_id": "stale", "expected_revision": 9}
            )
            assert bad.status_code == 409
            body = {"request_id": "forget", "expected_revision": 1}
            result = await client.post(path + "/forget", json=body)
            assert result.status_code == 200
            assert result.json()["source_message_id"] == str(source.id)
            assert result.json()["memory_id"] is None
            assert result.json()["status"] == "completed"
            assert (await client.post(path + "/forget", json=body)).json() == result.json()
            assert (await client.get(path + "/forget-preview")).status_code == 410


async def test_memory_selection_and_explanation_routes_require_auth_and_revision(conversation):
    from uuid import uuid4

    from hmph.api import create_app
    from hmph.config import Settings

    token = "test-only-token-at-least-32-characters"
    app = create_app(
        Settings(
            api_token=token,
            request_digest_key="test-only-digest-key-at-least-32-characters",
            _env_file=None,
        ),
        conversation=conversation,
    )
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            assert (await client.get("/api/memory-selection")).status_code == 401
            assert (
                await client.put(
                    "/api/memory-selection", json={"request_id": "select", "expected_revision": 1}
                )
            ).status_code == 401
            client.headers["Authorization"] = "Bearer " + token
            await client.post("/api/conversations", json={"request_id": "chat"})
            assert (await client.get("/api/memory-selection")).json() == {
                "mode": "lexical",
                "revision": 1,
                "build_id": None,
                "available": True,
            }
            selected = await client.put(
                "/api/memory-selection",
                json={"request_id": "select", "expected_revision": 1, "build_id": None},
            )
            assert selected.status_code == 200 and selected.json()["revision"] == 2
            assert (
                await client.put(
                    "/api/memory-selection", json={"request_id": "stale", "expected_revision": 1}
                )
            ).status_code == 409
            assert (
                await client.put(
                    "/api/memory-selection",
                    json={"request_id": "bad", "expected_revision": 2, "build_id": str(uuid4())},
                )
            ).json()["error"]["code"] == "memory_build_unavailable"
            assert (await client.get(f"/api/turns/{uuid4()}/memory-context")).status_code == 404


async def test_http_acceptance_and_cursor_replay_with_an_independent_worker(conversation):
    from hmph.api import create_app
    from hmph.config import Settings
    from hmph.model_gateway import FakeModel
    from hmph.worker import Worker

    settings = Settings(
        api_token="test-only-token-at-least-32-characters",
        request_digest_key="test-only-digest-key-at-least-32-characters",
        _env_file=None,
    )
    app = create_app(settings, conversation=conversation)
    headers = {"Authorization": "Bearer test-only-token-at-least-32-characters"}
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test", headers=headers
        ) as client:
            chat = (await client.post("/api/conversations", json={"request_id": "chat"})).json()
            accepted = await client.post(
                f"/api/conversations/{chat['id']}/turns",
                json={
                    "request_id": "turn",
                    "content": "你好",
                },
            )
            assert accepted.status_code == 202
            turn_id = accepted.json()["id"]
            assert (await client.get(f"/api/turns/{turn_id}")).json()["status"] == "accepted"
            await Worker(conversation, FakeModel()).run_once()
            replay = await client.get(
                f"/api/turns/{turn_id}/events", headers={"Last-Event-ID": "2"}
            )
            assert replay.status_code == 200
            assert "text/event-stream" in replay.headers["content-type"]
            assert "id: 1\n" not in replay.text and "id: 2\n" not in replay.text
            assert "id: 3\n" in replay.text and "event: turn.completed" in replay.text


async def test_api_requires_credentials_and_labels_fake_mode(conversation):
    from hmph.api import create_app
    from hmph.config import Settings

    app = create_app(
        Settings(
            api_token="test-only-token-at-least-32-characters",
            request_digest_key="test-only-digest-key-at-least-32-characters",
            _env_file=None,
        ),
        conversation=conversation,
    )
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            denied = await client.post("/api/conversations", json={"request_id": "chat"})
            assert denied.status_code == 401
            assert (await client.get("/health")).json()["model_mode"] == "fake"


@pytest.mark.parametrize("backend_name", ["mem0", "graphiti"])
async def test_framework_build_http_is_authenticated_idempotent_and_shadow_only(
    conversation, backend_name
):
    from hmph.api import create_app
    from hmph.config import Settings

    settings = Settings(
        api_token="test-only-token-at-least-32-characters",
        request_digest_key="test-only-digest-key-at-least-32-characters",
        _env_file=None,
    )
    app = create_app(settings, conversation=conversation)
    async with app.router.lifespan_context(app):
        identity = app.state.identity
        chat = await conversation.create_chat(identity.owner_id, identity.character_id, "chat")
        turn = await conversation.accept(identity.owner_id, chat.id, "fact", "我喜欢无糖茶。")
        await conversation.cancel(identity.owner_id, turn.id, "cancel")
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            assert (
                await client.post("/api/memory-builds", json={"request_id": "build"})
            ).status_code == 401
            client.headers["Authorization"] = "Bearer test-only-token-at-least-32-characters"
            body = {"request_id": "build", "backend": backend_name}
            created = await client.post("/api/memory-builds", json=body)
            assert created.status_code == 202
            build = created.json()
            assert build["source_count"] == 1 and build["mode"] == "shadow"
            assert "无糖茶" not in created.text
            assert (await client.post("/api/memory-builds", json=body)).json()["id"] == build["id"]
            assert (await client.get(f"/api/memory-builds/{build['id']}")).json()[
                "status"
            ] == "queued"
            assert (await client.get(f"/api/memory-builds/{build['id']}/spaces")).json() == []
            assert (await client.get("/api/memory-builds/current")).json() is None
            assert (await client.get("/health")).json()[
                "memory_status"
            ] == "sourced_lexical_baseline"


async def test_readiness_rejects_outdated_or_missing_configured_schema_version(conversation):
    from sqlalchemy import text

    from hmph.api import create_app
    from hmph.config import Settings

    app = create_app(
        Settings(
            api_token="test-only-token-at-least-32-characters",
            request_digest_key="test-only-digest-key-at-least-32-characters",
            _env_file=None,
        ),
        conversation=conversation,
    )
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            assert (await client.get("/ready")).status_code == 200
            # Fault injection touches only this test's schema.
            async with conversation.db.engine.begin() as connection:
                await connection.execute(text("UPDATE alembic_version SET version_num='0007'"))
            outdated = await client.get("/ready")
            assert outdated.status_code == 503
            assert outdated.json()["status"] == "database_migration_required"
            async with conversation.db.engine.begin() as connection:
                await connection.execute(
                    text(f'DROP TABLE "{conversation.db.schema}".alembic_version')
                )
            assert (await client.get("/ready")).status_code == 503
