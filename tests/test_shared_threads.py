from uuid import UUID, uuid4

import pytest

from hmph.contracts import DomainError
from hmph.memory import Memory
from hmph.model_gateway import SourceMention
from hmph.shared_threads import SharedThreads, interaction_rhythm


async def prepare(conversation, identity, chat, key, text="我回来了。"):
    await conversation.accept(identity.owner_id, chat.id, key, text)
    lease = await conversation.claim(str(uuid4()), 15)
    assert lease is not None
    return lease, await conversation.generation_request(lease)


@pytest.mark.parametrize(
    "query",
    ["我们现在还有什么共同计划？", "叫我什么、现在在哪工作、我们还在跟进什么计划？"],
)
async def test_explicit_plan_question_finds_active_thread_without_its_topic_word(
    conversation, query
):
    identity = await conversation.bootstrap()
    chat = await conversation.create_chat(identity.owner_id, identity.character_id, "plan")
    thread = await SharedThreads(conversation).write(
        identity.owner_id, chat.id, "create", topic="每周一起聊聊练琴进展"
    )
    next_chat = await conversation.create_chat(identity.owner_id, identity.character_id, "next")
    lease, request = await prepare(conversation, identity, next_chat, "ask", query)
    assert request.companion.shared_thread is not None
    assert not request.companion.no_active_shared_thread
    assert request.companion.shared_thread["topic"] == thread["topic"]
    assert any(str(s.message_id) == thread["source_message_id"] for s in request.raw_evidence)
    await conversation.finish(lease)
    receipt = await Memory(conversation.db, digest_key=conversation.digest_key).forget_source(
        identity.owner_id, UUID(thread["source_message_id"]), "forget", expected_revision=1
    )
    assert receipt.status == "completed"
    lease, request = await prepare(
        conversation, identity, next_chat, "forgotten", "我们现在还在一起跟进那个计划吗？"
    )
    assert request.companion.shared_thread is None
    assert request.companion.no_active_shared_thread
    assert thread["topic"] not in request.model_dump_json()
    await conversation.finish(lease)


async def test_shared_plan_survives_sessions_pauses_resumes_and_forgets_all_sources(conversation):
    identity = await conversation.bootstrap()
    chat = await conversation.create_chat(identity.owner_id, identity.character_id, "thread-home")
    service = SharedThreads(conversation)
    thread = await service.write(
        identity.owner_id, chat.id, "create-plan", topic="每周一起聊聊练琴进展"
    )
    assert (
        await service.write(identity.owner_id, chat.id, "create-plan", topic=thread["topic"])
        == thread
    )
    next_chat = await conversation.create_chat(identity.owner_id, identity.character_id, "next-day")
    lease, request = await prepare(conversation, identity, next_chat, "return")
    assert request.companion.shared_thread["topic"] == thread["topic"]
    assert any(str(s.message_id) == thread["source_message_id"] for s in request.raw_evidence)
    body = "回来啦，今天想聊聊练琴进展吗？"
    await conversation.publish(
        lease,
        0,
        body,
        mentions=(
            SourceMention(
                source_id=UUID(thread["source_message_id"]),
                quote="练琴进展",
                source_quote="练琴进展",
            ),
        ),
    )
    await conversation.finish(lease)
    thread = await service.write(
        identity.owner_id,
        next_chat.id,
        "pause",
        thread_id=UUID(thread["id"]),
        expected_revision=1,
        status="paused",
    )
    lease, request = await prepare(conversation, identity, next_chat, "paused-return")
    assert request.companion.shared_thread is None
    assert thread["topic"] not in request.model_dump_json()
    await conversation.finish(lease)
    thread = await service.write(
        identity.owner_id,
        next_chat.id,
        "resume",
        thread_id=UUID(thread["id"]),
        expected_revision=2,
        status="open",
    )
    lease, request = await prepare(conversation, identity, next_chat, "resumed-return")
    assert request.companion.shared_thread is not None
    await conversation.finish(lease)
    lease, request = await prepare(conversation, identity, next_chat, "closing", "晚安。")
    assert request.companion.act == "close" and request.companion.shared_thread is None
    await conversation.finish(lease)
    memory = Memory(conversation.db, digest_key=conversation.digest_key)
    preview = await memory.preview_forget_source(
        identity.owner_id, UUID(thread["source_message_id"])
    )
    receipt = await memory.forget_source(
        identity.owner_id,
        UUID(thread["source_message_id"]),
        "forget-plan",
        expected_revision=preview.revision,
    )
    assert receipt.status == "completed"
    assert (await service.read(identity.owner_id, identity.character_id))["items"] == []
    lease, request = await prepare(conversation, identity, next_chat, "forgotten-return")
    assert thread["topic"] not in request.model_dump_json()
    assert request.companion.shared_thread is None
    await conversation.finish(lease)


async def test_shared_plan_revision_ownership_capacity_and_inflight_invalidation(conversation):
    identity = await conversation.bootstrap()
    chat = await conversation.create_chat(identity.owner_id, identity.character_id, "controls")
    service = SharedThreads(conversation)
    first = await service.write(identity.owner_id, chat.id, "first", topic="一起练琴")
    lease, _ = await prepare(conversation, identity, chat, "active")
    await service.write(identity.owner_id, chat.id, "second", topic="一起读书")
    with pytest.raises(DomainError):
        await conversation.publish(lease, 0, "旧的建议。")
    with pytest.raises(DomainError, match="thread_revision_conflict"):
        await service.write(
            identity.owner_id,
            chat.id,
            "stale",
            thread_id=UUID(first["id"]),
            expected_revision=99,
            status="paused",
        )
    with pytest.raises(DomainError, match="not_found"):
        await service.write(uuid4(), chat.id, "wrong-owner", topic="不该创建")
    await service.write(identity.owner_id, chat.id, "third", topic="一起散步")
    with pytest.raises(DomainError, match="active_thread_limit"):
        await service.write(identity.owner_id, chat.id, "fourth", topic="第四件事")


def test_rhythm_counts_actual_questions_and_advice():
    rhythm = interaction_rhythm(["今天怎么样？", "我建议试试看，你想试吗？"])
    assert rhythm["recent_questions"] == 2 and rhythm["question_limit"] == 0
    assert rhythm["avoid_repeated_advice"] is True
    assert interaction_rhythm([])["question_limit"] == 1


async def test_explicit_chat_commands_apply_in_own_turn_without_mutating_from_hypotheses(
    conversation,
):
    identity = await conversation.bootstrap()
    chat = await conversation.create_chat(identity.owner_id, identity.character_id, "commands")
    service = SharedThreads(conversation)
    lease, request = await prepare(
        conversation, identity, chat, "command-create", "记下共同计划：一起练琴。"
    )
    assert request.companion.thread_control == "open"
    assert lease.policy_revision == 2
    await conversation.finish(lease)
    lease, request = await prepare(
        conversation, identity, chat, "command-pause", "请暂停共同计划：一起练琴。"
    )
    assert request.companion.thread_control == "paused"
    assert request.companion.shared_thread_action == "pause"
    assert "一起练琴" not in request.model_dump_json()
    await conversation.finish(lease)
    lease, request = await prepare(
        conversation, identity, chat, "command-hypothesis", "如果恢复共同计划：一起练琴。"
    )
    assert request.companion.thread_control is None
    await conversation.finish(lease)
    assert (await service.read(identity.owner_id, identity.character_id))["items"][0][
        "status"
    ] == "paused"
    lease, request = await prepare(
        conversation, identity, chat, "command-resume", "恢复共同计划：一起练琴。"
    )
    assert request.companion.thread_control == "open" and lease.policy_revision == 4
    assert request.companion.shared_thread is not None
    await conversation.finish(lease)


async def test_shared_plan_http_management_and_cookie_write_protection(conversation):
    import httpx

    from hmph.api import create_app
    from hmph.config import Settings

    token = "shared-thread-fixture-token-at-least-32-characters"
    app = create_app(
        Settings(
            api_token=token,
            request_digest_key="fixture-request-digest-at-least-32-characters",
            _env_file=None,
        ),
        conversation=conversation,
    )
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://127.0.0.1:8017"
        ) as client:
            origin = {"Origin": "http://127.0.0.1:8017"}
            login = await client.post("/api/session", json={"token": token}, headers=origin)
            headers = {**origin, "X-CSRF-Token": login.json()["csrf_token"]}
            chat = (
                await client.post(
                    "/api/conversations", json={"request_id": "http-chat"}, headers=headers
                )
            ).json()
            body = {"conversation_id": chat["id"], "request_id": "http-create", "topic": "一起读书"}
            assert (
                await client.post("/api/shared-threads", json=body, headers=origin)
            ).status_code == 403
            created = await client.post("/api/shared-threads", json=body, headers=headers)
            assert created.status_code == 201
            item = created.json()
            sources = await client.get(f"/api/shared-threads/{item['id']}/sources")
            assert sources.status_code == 200
            assert "一起读书" in sources.json()["items"][0]["content"]
            assert (await client.get("/api/shared-threads")).json()["items"][0]["id"] == item["id"]
            response = await client.post(
                f"/api/shared-threads/{item['id']}",
                headers=headers,
                json={
                    "conversation_id": chat["id"],
                    "request_id": "http-pause",
                    "expected_revision": item["revision"],
                    "status": "paused",
                },
            )
            assert response.status_code == 200 and response.json()["status"] == "paused"
