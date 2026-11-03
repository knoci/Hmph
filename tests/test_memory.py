from hmph.model_gateway import FakeModel
from hmph.worker import Worker


async def test_dated_episodes_keep_distinct_occurrences_and_recall_requested_month(conversation):
    from hmph.memory import Memory
    from hmph.memory_types import MemoryProposal

    identity = await conversation.bootstrap()
    memory = Memory(conversation.db, digest_key=conversation.digest_key)
    for index, month in enumerate((2, 5)):
        text = f"2024年{month}月我去了京都。"
        await say(conversation, identity, text, f"trip-{index}")
        await memory.learn(
            await memory.claim("extractor"),
            [
                MemoryProposal(
                    kind="episode",
                    predicate="trip",
                    value="京都",
                    quote=text,
                    scope="historical",
                    time_text=f"2024年{month}月",
                )
            ],
        )
    entries = await memory.entries(identity.owner_id, identity.character_id)
    assert len(entries) == 2
    hits = await memory.recall(identity.owner_id, identity.character_id, "2024年2月京都那次旅行")
    assert len(hits) == 1 and hits[0].time_text == "2024年2月"
    assert hits[0].referenced_time.precision == "month"
    assert hits[0].referenced_time.start.isoformat() == "2024-02-01T00:00:00+00:00"
    assert hits[0].referenced_time.end.isoformat() == "2024-03-01T00:00:00+00:00"
    assert not await memory.recall(
        identity.owner_id, identity.character_id, "2024年3月京都那次旅行"
    )
    await memory.forget(identity.owner_id, hits[0].id, "forget-february", expected_revision=1)
    remaining = await memory.recall(
        identity.owner_id, identity.character_id, "2024年5月京都那次旅行"
    )
    assert len(remaining) == 1 and remaining[0].time_text == "2024年5月"


async def say(conversation, identity, text, request_id):
    chat = await conversation.create_chat(identity.owner_id, identity.character_id, request_id)
    turn = await conversation.accept(identity.owner_id, chat.id, request_id, text)
    await conversation.cancel(identity.owner_id, turn.id, request_id)
    return (await conversation.messages(identity.owner_id, chat.id))[0]


async def test_user_fact_is_persisted_with_source_and_recalled_across_sessions(conversation):
    from hmph.memory import Memory
    from hmph.memory_types import MemoryProposal

    identity = await conversation.bootstrap()
    chat = await conversation.create_chat(identity.owner_id, identity.character_id, "chat")
    await conversation.accept(identity.owner_id, chat.id, "turn", "我叫小沐。")
    await Worker(conversation, FakeModel()).run_once()
    source = (await conversation.messages(identity.owner_id, chat.id))[0]

    memory = Memory(conversation.db, digest_key=conversation.digest_key)
    task = await memory.claim("memory-worker", model_mode="fake")
    result = await memory.learn(
        task, [MemoryProposal(predicate="name", value="小沐", quote="我叫小沐")]
    )
    hits = await memory.recall(identity.owner_id, identity.character_id, "我叫什么？")

    assert result.status == "committed"
    assert result.written == 1
    assert len(hits) == 1 and hits[0].value == "小沐"
    assert hits[0].sources[0].message_id == source.id
    assert hits[0].sources[0].message_revision == 1
    assert hits[0].recorded_at is not None


async def test_correction_keeps_prior_sources_but_recall_uses_latest_fact(conversation):
    from hmph.memory import Memory
    from hmph.memory_types import MemoryProposal

    identity = await conversation.bootstrap()
    memory = Memory(conversation.db, digest_key=conversation.digest_key)
    first = await say(conversation, identity, "我在星河公司工作。", "old")
    lease = await memory.claim("extractor")
    proposal = MemoryProposal(predicate="employer", value="星河公司", quote="我在星河公司工作")
    await memory.learn(lease, [proposal])
    original = (await memory.recall(identity.owner_id, identity.character_id, "工作"))[0]
    second = await say(conversation, identity, "现在我在远山公司工作。", "new")
    lease = await memory.claim("extractor")
    await memory.learn(
        lease,
        [
            MemoryProposal(
                predicate="employer",
                value="远山公司",
                quote="现在我在远山公司工作",
            )
        ],
    )
    hit = (await memory.recall(identity.owner_id, identity.character_id, "工作"))[0]
    assert hit.id == original.id and hit.revision == 2 and hit.value == "远山公司"
    history = await memory.history(identity.owner_id, hit.id)
    assert [v.value for v in history] == ["星河公司", "远山公司"]
    assert history[0].sources[0].message_id == first.id
    assert history[1].sources[0].message_id == second.id
    assert (await memory.learn(lease, [proposal])).status == "stale"


async def test_new_chat_receives_sourced_memory_without_old_chat_transcript(conversation):
    from hmph.memory import Memory
    from hmph.memory_types import MemoryProposal

    identity = await conversation.bootstrap()
    source = await say(conversation, identity, "我叫小沐。独立原话不要整段传入新会话。", "old")
    memory = Memory(conversation.db, digest_key=conversation.digest_key)
    await memory.learn(
        await memory.claim("extractor"),
        [
            MemoryProposal(predicate="name", value="小沐", quote="我叫小沐"),
        ],
    )
    new = await conversation.create_chat(identity.owner_id, identity.character_id, "new")
    await conversation.accept(identity.owner_id, new.id, "question", "我叫什么？")
    request = await conversation.generation_request(await conversation.claim("generator"))
    assert len(request.messages) == 1
    assert request.evidence[0].value == "小沐"
    assert request.evidence[0].sources[0].message_id == source.id
    assert "独立原话" not in request.model_dump_json()


async def test_forgetting_erases_sources_and_derived_replies_and_fences_old_workers(conversation):
    import pytest

    from hmph.contracts import DomainError
    from hmph.memory import Memory
    from hmph.memory_types import MemoryProposal

    identity = await conversation.bootstrap()
    memory = Memory(conversation.db, digest_key=conversation.digest_key)
    old_chat = await conversation.create_chat(identity.owner_id, identity.character_id, "old")
    await conversation.accept(identity.owner_id, old_chat.id, "fact", "我叫小沐。")
    await Worker(conversation, FakeModel()).run_once()
    await memory.learn(
        await memory.claim("extractor"),
        [
            MemoryProposal(predicate="name", value="小沐", quote="我叫小沐"),
        ],
    )
    hit = (await memory.recall(identity.owner_id, identity.character_id, "名字"))[0]

    new_chat = await conversation.create_chat(identity.owner_id, identity.character_id, "new")
    derived = await conversation.accept(identity.owner_id, new_chat.id, "question", "我叫什么？")
    lease = await conversation.claim("generator")
    await conversation.generation_request(lease)
    await conversation.publish(lease, 0, "你叫小沐。")
    await conversation.finish(lease)
    pending = await conversation.accept(identity.owner_id, new_chat.id, "next", "你还记得吗？")
    live_lease = await conversation.claim("generator")
    await conversation.generation_request(live_lease)
    old_extractor = await memory.claim("slow-extractor")

    preview = await memory.preview_forget(identity.owner_id, hit.id)
    assert preview.memories == 1 and preview.messages >= 3
    receipt = await memory.forget(identity.owner_id, hit.id, "forget", expected_revision=1)
    assert receipt.status == "completed"
    assert (
        await memory.forget(identity.owner_id, hit.id, "forget", expected_revision=1)
    ).id == receipt.id
    assert not await memory.recall(identity.owner_id, identity.character_id, "小沐名字")
    with pytest.raises(DomainError, match="memory_deleted"):
        await memory.history(identity.owner_id, hit.id)
    assert all(m.deleted for m in await conversation.messages(identity.owner_id, old_chat.id))
    messages = await conversation.messages(identity.owner_id, new_chat.id)
    assert messages[0].content == "我叫什么？" and messages[1].deleted
    assert "小沐" not in str(await conversation.events(identity.owner_id, derived.id))
    assert (await conversation.get_turn(identity.owner_id, pending.id)).status == "invalidated"
    with pytest.raises(DomainError, match="stale_lease"):
        await conversation.publish(live_lease, 0, "你叫小沐。")
    assert (
        await memory.learn(
            old_extractor,
            [
                MemoryProposal(predicate="name", value="什么", quote="我叫什么"),
            ],
        )
    ).status == "stale"


async def test_expired_extraction_is_retried_once_without_accepting_old_result(conversation):
    import asyncio

    from hmph.memory import Memory
    from hmph.memory_types import MemoryProposal
    from hmph.memory_worker import MemoryWorker

    identity = await conversation.bootstrap()
    await say(conversation, identity, "我叫小沐。", "fact")
    memory = Memory(conversation.db, digest_key=conversation.digest_key)
    old = await memory.claim("lost-worker", lease_seconds=0.01)
    await asyncio.sleep(0.02)
    calls = []

    async def extract(content):
        calls.append(content)
        return [MemoryProposal(predicate="name", value="小沐", quote="我叫小沐")], None

    worker = MemoryWorker(memory, extract)
    assert await worker.run_once()
    assert not await worker.run_once()
    assert calls == ["我叫小沐。"]
    assert (await memory.learn(old, [])).status == "stale"
    hit = (await memory.recall(identity.owner_id, identity.character_id, "名字"))[0]
    assert len(hit.sources) == 1


async def test_memory_management_api_requires_auth_and_returns_completed_deletion(conversation):
    import httpx

    from hmph.api import create_app
    from hmph.config import Settings
    from hmph.memory import Memory
    from hmph.memory_types import MemoryProposal

    identity = await conversation.bootstrap()
    source = await say(conversation, identity, "我叫小沐。", "fact")
    memory = Memory(conversation.db, digest_key=conversation.digest_key)
    await memory.learn(
        await memory.claim("extractor"),
        [
            MemoryProposal(predicate="name", value="小沐", quote="我叫小沐"),
        ],
    )
    token = "test-token-at-least-thirty-two-characters"
    app = create_app(
        Settings(api_token=token, request_digest_key=token, model_mode="fake", _env_file=None),
        conversation=conversation,
    )
    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://test",
        ) as client,
    ):
        assert (await client.get("/api/memories")).status_code == 401
        client.headers["Authorization"] = "Bearer " + token
        listed = await client.get("/api/memories", params={"query": "名字"})
        assert listed.status_code == 200
        entry = listed.json()[0]
        history = await client.get(f"/api/memories/{entry['id']}/history")
        assert history.json()[0]["sources"][0]["message_id"] == str(source.id)
        assert (await client.get(f"/api/memory-tasks/{source.id}")).json()["status"] == "done"
        preview = await client.get(f"/api/memories/{entry['id']}/forget-preview")
        assert preview.json()["messages"] == 1
        forgotten = await client.post(
            f"/api/memories/{entry['id']}/forget",
            json={
                "request_id": "forget",
                "expected_revision": 1,
            },
        )
        assert forgotten.status_code == 200 and forgotten.json()["status"] == "completed"
        assert (await client.get("/api/memories")).json() == []


async def test_past_question_recalls_old_version_instead_of_current_employer(conversation):
    from hmph.memory import Memory
    from hmph.memory_types import MemoryProposal

    identity = await conversation.bootstrap()
    memory = Memory(conversation.db, digest_key=conversation.digest_key)
    for index, company in enumerate(("星河公司", "远山公司")):
        await say(conversation, identity, f"我在{company}工作。", f"job-{index}")
        await memory.learn(
            await memory.claim("extractor"),
            [
                MemoryProposal(
                    predicate="employer",
                    value=company,
                    quote=f"我在{company}工作",
                )
            ],
        )
    past = await memory.recall(identity.owner_id, identity.character_id, "我以前在哪家公司工作？")
    assert len(past) == 1 and past[0].value == "星河公司" and past[0].scope == "historical"
    current = await memory.recall(
        identity.owner_id, identity.character_id, "我现在在哪家公司工作？"
    )
    assert len(current) == 1 and current[0].value == "远山公司"


async def test_late_old_source_cannot_replace_newer_current_fact(conversation):
    from hmph.memory import Memory
    from hmph.memory_types import MemoryProposal

    identity = await conversation.bootstrap()
    await say(conversation, identity, "我在星河公司工作。", "old")
    chat = await conversation.create_chat(identity.owner_id, identity.character_id, "new")
    turn = await conversation.accept(
        identity.owner_id, chat.id, "new-job", "我在远山公司工作。", model_mode="deepseek"
    )
    await conversation.cancel(identity.owner_id, turn.id, "cancel")
    memory = Memory(conversation.db, digest_key=conversation.digest_key)
    newer = await memory.claim("extractor", model_mode="deepseek")
    await memory.learn(
        newer, [MemoryProposal(predicate="employer", value="远山公司", quote="我在远山公司工作")]
    )
    older = await memory.claim("extractor")
    await memory.learn(
        older, [MemoryProposal(predicate="employer", value="星河公司", quote="我在星河公司工作")]
    )
    assert [
        m.value for m in await memory.recall(identity.owner_id, identity.character_id, "现在的公司")
    ] == ["远山公司"]
    assert [
        m.value for m in await memory.recall(identity.owner_id, identity.character_id, "以前的公司")
    ] == ["星河公司"]


async def test_unproven_proposal_is_rejected_atomically_and_other_owner_cannot_read_or_forget(
    conversation,
):
    from uuid import uuid4

    import pytest

    from hmph.contracts import DomainError
    from hmph.memory import Memory
    from hmph.memory_types import MemoryProposal

    identity = await conversation.bootstrap()
    outsider = await conversation.bootstrap(uuid4())
    await say(conversation, identity, "我叫小沐。", "fact")
    memory = Memory(conversation.db, digest_key=conversation.digest_key)
    lease = await memory.claim("extractor")
    valid = MemoryProposal(predicate="name", value="小沐", quote="我叫小沐")
    with pytest.raises(DomainError, match="unsupported_memory"):
        await memory.learn(
            lease, [valid, MemoryProposal(predicate="home_city", value="杭州", quote="住在杭州")]
        )
    assert not await memory.entries(identity.owner_id, identity.character_id)
    await memory.learn(lease, [valid])
    hit = (await memory.recall(identity.owner_id, identity.character_id, "名字"))[0]
    assert not await memory.recall(outsider.owner_id, identity.character_id, "小沐名字")
    with pytest.raises(DomainError, match="not_found"):
        await memory.history(outsider.owner_id, hit.id)
    with pytest.raises(DomainError, match="not_found"):
        await memory.forget(outsider.owner_id, hit.id, "forget", expected_revision=1)
    with pytest.raises(DomainError, match="memory_revision_conflict"):
        await memory.forget(identity.owner_id, hit.id, "forget", expected_revision=2)
    assert (await memory.recall(identity.owner_id, identity.character_id, "名字"))[
        0
    ].value == "小沐"


async def test_credentials_and_explicit_nonretention_never_reach_extractor(conversation):
    from hmph.memory import Memory
    from hmph.memory_worker import MemoryWorker

    identity = await conversation.bootstrap()
    for index, text in enumerate(
        ("我叫小沐，但不要记住。", "测试 key sk-this-is-a-fake-credential-for-tests")
    ):
        await say(conversation, identity, text, f"source-{index}")
    calls = []

    async def extract(content):
        calls.append(content)
        return [], None

    memory = Memory(conversation.db, digest_key=conversation.digest_key)
    worker = MemoryWorker(memory, extract)
    assert not await worker.run_once()
    assert not await worker.run_once()
    assert not calls


async def test_forget_also_removes_unprocessed_literal_copies_from_recent_history(conversation):
    from uuid import uuid4

    from hmph.memory import Memory
    from hmph.memory_types import MemoryProposal

    identity = await conversation.bootstrap()
    memory = Memory(conversation.db, digest_key=conversation.digest_key)
    await say(conversation, identity, "我叫小沐。", "first")
    await memory.learn(
        await memory.claim("extractor"),
        [
            MemoryProposal(predicate="name", value="小沐", quote="我叫小沐"),
        ],
    )
    entry = (await memory.recall(identity.owner_id, identity.character_id, "名字"))[0]
    copy = await say(conversation, identity, "再说一遍，我叫小沐。", "copy")
    outstanding = await memory.claim("slow-extractor")
    outsider = await conversation.bootstrap(uuid4())
    other_source = await say(conversation, outsider, "我也叫小沐。", "other")

    preview = await memory.preview_forget(identity.owner_id, entry.id)
    assert preview.messages == 2
    await memory.forget(identity.owner_id, entry.id, "forget", expected_revision=1)
    assert (
        await memory.learn(
            outstanding,
            [
                MemoryProposal(predicate="name", value="小沐", quote="我叫小沐"),
            ],
        )
    ).status == "stale"
    owner_chats = await conversation.chats(identity.owner_id)
    remaining = [
        m for c in owner_chats for m in await conversation.messages(identity.owner_id, c.id)
    ]
    assert next(m for m in remaining if m.id == copy.id).deleted
    other_chats = await conversation.chats(outsider.owner_id)
    assert (await conversation.messages(outsider.owner_id, other_chats[0].id))[
        0
    ].id == other_source.id
    assert not (await conversation.messages(outsider.owner_id, other_chats[0].id))[0].deleted


async def test_repeated_confirmations_do_not_crowd_fact_out_of_context_budget(conversation):
    from hmph.memory import Memory
    from hmph.memory_types import MemoryProposal

    identity = await conversation.bootstrap()
    memory = Memory(conversation.db, digest_key=conversation.digest_key)
    for index in range(20):
        await say(conversation, identity, "我叫小沐。", f"confirmation-{index}")
        await memory.learn(
            await memory.claim("extractor"),
            [
                MemoryProposal(predicate="name", value="小沐", quote="我叫小沐"),
            ],
        )
    chat = await conversation.create_chat(identity.owner_id, identity.character_id, "question")
    await conversation.accept(identity.owner_id, chat.id, "question", "我叫什么？")
    request = await conversation.generation_request(await conversation.claim("generator"))
    assert request.evidence and request.evidence[0].value == "小沐"
    full = await memory.history(identity.owner_id, request.evidence[0].id)
    assert len(full[0].sources) == 20
    assert request.evidence[0].total_sources == 20
    assert len(request.evidence[0].sources) <= 2


async def test_forget_original_without_extracted_entry_clears_replies_and_stale_work(conversation):
    from uuid import uuid4

    import pytest

    from hmph.contracts import DomainError
    from hmph.memory import Memory
    from hmph.memory_types import MemoryProposal

    identity = await conversation.bootstrap()
    memory = Memory(conversation.db, digest_key=conversation.digest_key)
    chat = await conversation.create_chat(identity.owner_id, identity.character_id, "original")
    turn = await conversation.accept(
        identity.owner_id, chat.id, "fact", "我买的折叠琴是 Moonbell。"
    )
    await conversation.cancel(identity.owner_id, turn.id, "cancel")
    original = (await conversation.messages(identity.owner_id, chat.id))[0]
    extraction = await memory.claim("slow-extractor")
    other = await conversation.bootstrap(uuid4())
    other_chat = await conversation.create_chat(other.owner_id, other.character_id, "other")
    other_turn = await conversation.accept(
        other.owner_id, other_chat.id, "fact", "我买的折叠琴是 Moonbell。"
    )
    await conversation.cancel(other.owner_id, other_turn.id, "cancel")
    query_chat = await conversation.create_chat(identity.owner_id, identity.character_id, "query")
    await conversation.accept(identity.owner_id, query_chat.id, "query", "我上次买的折叠琴是什么？")
    lease = await conversation.claim("generator")
    req = await conversation.generation_request(lease)
    assert original.id in {s.message_id for s in req.raw_evidence}
    await conversation.publish(lease, 0, "Moonbell。")
    await conversation.finish(lease)
    assert not await memory.entries(identity.owner_id, identity.character_id)
    preview = await memory.preview_forget_source(identity.owner_id, original.id)
    assert preview.memory_id is None and preview.source_message_id == original.id
    assert preview.memories == 0 and preview.messages == 2
    with pytest.raises(DomainError, match="source_revision_conflict"):
        await memory.forget_source(identity.owner_id, original.id, "bad", expected_revision=99)
    with pytest.raises(DomainError, match="not_found"):
        await memory.preview_forget_source(other.owner_id, original.id)
    receipt = await memory.forget_source(
        identity.owner_id, original.id, "forget", expected_revision=1
    )
    assert receipt.memory_id is None and receipt.source_message_id == original.id
    assert receipt.status == "completed"
    assert (
        await memory.forget_source(identity.owner_id, original.id, "forget", expected_revision=1)
    ).id == receipt.id
    assert (
        await memory.learn(
            extraction,
            [
                MemoryProposal(
                    predicate="instrument", value="Moonbell", quote="我买的折叠琴是 Moonbell。"
                )
            ],
        )
    ).status == "stale"
    assert all(
        m.deleted
        for m in await conversation.messages(identity.owner_id, query_chat.id)
        if m.role == "assistant"
    )
    assert not (await conversation.messages(other.owner_id, other_chat.id))[0].deleted
