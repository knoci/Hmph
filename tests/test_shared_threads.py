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
    assert thread["topic"] in request.model_dump_json()
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


async def test_named_status_uses_its_plan_evidence_and_keeps_mixed_questions(conversation):
    identity = await conversation.bootstrap()
    owner = identity.owner_id
    chat = await conversation.create_chat(owner, identity.character_id, "mixed-history")
    unrelated = "我还得周四中午去青禾木工房取胸针。"
    lease, _ = await prepare(conversation, identity, chat, "pickup", unrelated)
    await conversation.publish(lease, 0, "好，周四中午取胸针。")
    await conversation.finish(lease)

    threads = SharedThreads(conversation)
    plan = await threads.write(owner, chat.id, "make-plan", topic="周末的木刻小集试做")
    await threads.write(
        owner, chat.id, "pause-plan", thread_id=UUID(plan["id"]), expected_revision=1,
        status="paused",
    )
    lease, request = await prepare(
        conversation, identity, chat, "named-status", "木刻小集的试做现在是什么状态？"
    )
    assert request.companion.shared_thread["status"] == "paused"
    assert "青禾木工房" not in request.model_dump_json()
    used_source_ids = {str(source.message_id) for source in request.raw_evidence}
    assert plan["source_message_id"] in used_source_ids
    await conversation.finish(lease)

    lease, request = await prepare(
        conversation, identity, chat, "mixed-status", f"{plan['topic']}还在继续吗？胸针什么时候取？"
    )
    assert request.companion.shared_thread["status"] == "paused"
    assert any(message.content == unrelated for message in request.messages)
    await conversation.finish(lease)


async def test_plan_lookup_prefers_the_named_plan_and_lists_multiple_active_plans(conversation):
    identity = await conversation.bootstrap()
    owner = identity.owner_id
    chat = await conversation.create_chat(owner, identity.character_id, "plan-list")
    service = SharedThreads(conversation)
    plans = [
        await service.write(owner, chat.id, f"create-{i}", topic=topic)
        for i, topic in enumerate(("盒子里整理老照片", "学会完整弹奏月光奏鸣曲"))
    ]
    first, named = sorted(plans, key=lambda plan: plan["id"])
    reopened = await conversation.create_chat(owner, identity.character_id, "plan-reopened")
    lease, request = await prepare(
        conversation, identity, reopened, "named", f"还记得「{named['topic']}」那个共同计划吗？"
    )
    assert request.companion.shared_thread["id"] == named["id"]
    await conversation.finish(lease)

    lease, request = await prepare(
        conversation, identity, reopened, "list", "我们还有哪些共同计划？"
    )
    assert request.companion.shared_thread is None
    offered = request.companion.model_context()["shared_threads"]
    assert {thread["id"] for thread in offered} == {first["id"], named["id"]}
    assert request.companion.shared_threads_complete
    assert {t["source_id"] for t in offered} <= {str(s.message_id) for s in request.raw_evidence}
    await conversation.finish(lease)

    await service.write(
        owner, chat.id, "pause", thread_id=UUID(named["id"]), expected_revision=1, status="paused"
    )
    lease, request = await prepare(
        conversation, identity, reopened, "list-open", "现在还在继续的共同计划还有哪些？"
    )
    assert request.companion.shared_threads_complete
    assert request.companion.shared_thread["id"] == first["id"]
    assert request.companion.shared_threads == ()
    assert named["source_message_id"] not in {str(s.message_id) for s in request.raw_evidence}
    await conversation.finish(lease)

    lease, request = await prepare(
        conversation,
        identity,
        reopened,
        "paused-by-name",
        f"「{named['topic']}」共同计划怎么样了？",
    )
    assert request.companion.shared_thread["id"] == named["id"]
    assert request.companion.shared_thread["status"] == "paused"
    assert not request.companion.shared_threads_complete
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
    assert rhythm == {"recent_questions": 2, "recent_advice": 1}
    assert interaction_rhythm([]) == {"recent_questions": 0, "recent_advice": 0}


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
    assert request.messages[-1].content == "请暂停共同计划：一起练琴。"
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
