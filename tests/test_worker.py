from hmph.model_gateway import ModelDelta


async def test_worker_publishes_durable_segments_and_replay_skips_seen_events(conversation):
    from hmph.model_gateway import FakeModel
    from hmph.worker import Worker

    identity = await conversation.bootstrap()
    chat = await conversation.create_chat(identity.owner_id, identity.character_id, "chat")
    turn = await conversation.accept(identity.owner_id, chat.id, "turn", "你好")
    assert await Worker(conversation, FakeModel()).run_once()

    saved = await conversation.get_turn(identity.owner_id, turn.id)
    events = await conversation.events(identity.owner_id, turn.id)
    replay = await conversation.events(identity.owner_id, turn.id, after=2)
    messages = await conversation.messages(identity.owner_id, chat.id)

    assert saved.status == "completed"
    assert [event.type for event in events] == [
        "turn.accepted",
        "text.delta",
        "text.delta",
        "turn.completed",
    ]
    assert [event.id for event in replay] == [3, 4]
    assert messages[-1].content == "哼，收到啦。我现在使用的是测试模型，还没有接入真实记忆。"


async def test_expired_worker_cannot_resume_and_partial_output_is_not_regenerated(conversation):
    import asyncio

    import pytest

    from hmph.contracts import DomainError
    from hmph.model_gateway import FakeModel
    from hmph.worker import Worker

    identity = await conversation.bootstrap()
    chat = await conversation.create_chat(identity.owner_id, identity.character_id, "chat")
    turn = await conversation.accept(identity.owner_id, chat.id, "turn", "你好")
    lease = await conversation.claim("crashed-worker", lease_seconds=0.05)
    first = await conversation.publish(lease, 0, "哼，收到啦。")
    # A lost publish acknowledgement is retried with the same segment number.
    assert (await conversation.publish(lease, 0, "哼，收到啦。")).id == first.id
    await asyncio.sleep(0.08)

    assert await conversation.recover_expired() == 1
    assert not await Worker(conversation, FakeModel()).run_once()
    with pytest.raises(DomainError, match="stale_lease"):
        await conversation.publish(lease, 1, "旧执行者恢复了。")
    saved = await conversation.get_turn(identity.owner_id, turn.id)
    assert (saved.status, saved.error_code) == ("failed", "worker_lost")
    assert [
        message.content for message in await conversation.messages(identity.owner_id, chat.id)
    ] == [
        "你好",
        "哼，收到啦。",
    ]


async def test_slow_provider_keeps_lease_alive(conversation):
    from hmph.model_gateway import FakeModel
    from hmph.worker import Worker

    identity = await conversation.bootstrap()
    chat = await conversation.create_chat(identity.owner_id, identity.character_id, "chat")
    turn = await conversation.accept(identity.owner_id, chat.id, "turn", "你好")
    await Worker(conversation, FakeModel(delay=0.15), lease_seconds=0.12).run_once()
    assert (await conversation.get_turn(identity.owner_id, turn.id)).status == "completed"


async def test_cancel_during_generation_stops_later_commits(conversation):
    import asyncio

    from hmph.model_gateway import FakeModel
    from hmph.worker import Worker

    identity = await conversation.bootstrap()
    chat = await conversation.create_chat(identity.owner_id, identity.character_id, "chat")
    turn = await conversation.accept(identity.owner_id, chat.id, "turn", "你好")
    running = asyncio.create_task(Worker(conversation, FakeModel(delay=0.1)).run_once())
    async with asyncio.timeout(3):
        # Poll the public persisted event log, as a client in another process would.
        while len(await conversation.events(identity.owner_id, turn.id)) < 2:  # noqa: ASYNC110
            await asyncio.sleep(0.01)
        await conversation.cancel(identity.owner_id, turn.id, "cancel")
        await running
    events = await conversation.events(identity.owner_id, turn.id)
    assert [event.type for event in events] == [
        "turn.accepted",
        "text.delta",
        "turn.cancelled",
    ]


async def test_unapproved_draft_is_replaced_before_publication(conversation):
    from hmph.worker import Worker

    class UnapprovedProvider:
        async def stream(self, user_content):
            yield ModelDelta(text="这段未审核草稿不能发布。")

    identity = await conversation.bootstrap()
    chat = await conversation.create_chat(identity.owner_id, identity.character_id, "chat")
    turn = await conversation.accept(identity.owner_id, chat.id, "turn", "你好")
    await Worker(conversation, UnapprovedProvider()).run_once()
    events = await conversation.events(identity.owner_id, turn.id)
    assert [event.payload["text"] for event in events if event.type == "text.delta"] == [
        "这条回复暂时无法提供，请稍后再试。",
    ]


async def test_provider_failure_has_terminal_event_without_exception_details(conversation):
    from hmph.worker import Worker

    class BrokenProvider:
        async def stream(self, user_content):
            yield ModelDelta(text="哼，收到啦。")
            raise RuntimeError("private-provider-token-123")

    identity = await conversation.bootstrap()
    chat = await conversation.create_chat(identity.owner_id, identity.character_id, "chat")
    turn = await conversation.accept(identity.owner_id, chat.id, "turn", "你好")
    await Worker(conversation, BrokenProvider()).run_once()
    saved = await conversation.get_turn(identity.owner_id, turn.id)
    events = await conversation.events(identity.owner_id, turn.id)
    assert (saved.status, saved.error_code) == ("failed", "model_error")
    assert events[-1].type == "turn.failed"
    assert "private-provider-token" not in str(events)


async def test_empty_provider_response_is_a_visible_failure(conversation):
    from hmph.worker import Worker

    class EmptyProvider:
        async def stream(self, user_content):
            if False:
                yield ""

    identity = await conversation.bootstrap()
    chat = await conversation.create_chat(identity.owner_id, identity.character_id, "chat")
    turn = await conversation.accept(identity.owner_id, chat.id, "turn", "你好")
    await Worker(conversation, EmptyProvider()).run_once()
    saved = await conversation.get_turn(identity.owner_id, turn.id)
    assert (saved.status, saved.error_code) == ("failed", "empty_model_output")


async def test_two_workers_cannot_both_generate_the_same_turn(conversation):
    import asyncio

    from hmph.model_gateway import FakeModel
    from hmph.worker import Worker

    identity = await conversation.bootstrap()
    chat = await conversation.create_chat(identity.owner_id, identity.character_id, "chat")
    turn = await conversation.accept(identity.owner_id, chat.id, "turn", "你好")
    await asyncio.gather(
        *[Worker(conversation, FakeModel(delay=0.04)).run_once() for _ in range(2)]
    )
    events = await conversation.events(identity.owner_id, turn.id)
    assert [event.type for event in events].count("turn.completed") == 1
    assert [event.type for event in events].count("text.delta") == 2


async def test_expired_queued_input_fails_without_starting_generation(conversation):
    import asyncio

    from hmph.conversation import Conversation
    from hmph.model_gateway import FakeModel
    from hmph.worker import Worker

    short = Conversation(conversation.db, digest_key=b"test-key", turn_timeout=0.04)
    identity = await short.bootstrap()
    chat = await short.create_chat(identity.owner_id, identity.character_id, "chat")
    turn = await short.accept(identity.owner_id, chat.id, "turn", "你好")
    await asyncio.sleep(0.06)
    await Worker(short, FakeModel()).run_once()
    saved = await short.get_turn(identity.owner_id, turn.id)
    assert (saved.status, saved.error_code) == ("failed", "turn_timeout")
