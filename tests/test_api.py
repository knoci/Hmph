import httpx


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


async def test_framework_build_http_is_authenticated_idempotent_and_shadow_only(conversation):
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
            body = {"request_id": "build", "backend": "mem0"}
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
