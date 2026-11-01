async def test_accepted_turn_survives_new_service_instance(conversation):
    from hmph.conversation import Conversation

    identity = await conversation.bootstrap()
    chat = await conversation.create_chat(identity.owner_id, identity.character_id, "chat-1")
    turn = await conversation.accept(identity.owner_id, chat.id, "request-1", "今天开始做项目。")

    reopened = Conversation(conversation.db, digest_key=b"test-only-request-digest-key")
    saved = await reopened.get_turn(identity.owner_id, turn.id)
    messages = await reopened.messages(identity.owner_id, chat.id)
    events = await reopened.events(identity.owner_id, turn.id)

    assert saved.status == "accepted"
    assert [message.content for message in messages] == ["今天开始做项目。"]
    assert [(event.id, event.type) for event in events] == [(1, "turn.accepted")]


async def test_concurrent_retry_creates_one_turn_and_one_user_message(conversation):
    import asyncio

    identity = await conversation.bootstrap()
    chat = await conversation.create_chat(identity.owner_id, identity.character_id, "chat-1")
    turns = await asyncio.gather(
        *[conversation.accept(identity.owner_id, chat.id, "same-request", "你好") for _ in range(4)]
    )
    assert len({turn.id for turn in turns}) == 1
    assert len(await conversation.messages(identity.owner_id, chat.id)) == 1


async def test_cancel_is_durable_and_allows_the_next_turn(conversation):
    identity = await conversation.bootstrap()
    chat = await conversation.create_chat(identity.owner_id, identity.character_id, "chat-1")
    turn = await conversation.accept(identity.owner_id, chat.id, "turn-1", "先等等。")
    cancelled = await conversation.cancel(identity.owner_id, turn.id, "cancel-1")
    repeated = await conversation.cancel(identity.owner_id, turn.id, "cancel-1")
    events = await conversation.events(identity.owner_id, turn.id)
    next_turn = await conversation.accept(identity.owner_id, chat.id, "turn-2", "现在继续。")

    assert cancelled.status == repeated.status == "cancelled"
    assert [event.type for event in events] == ["turn.accepted", "turn.cancelled"]
    assert next_turn.status == "accepted"


async def test_same_key_with_changed_input_or_target_is_rejected(conversation):
    import pytest

    from hmph.contracts import DomainError

    identity = await conversation.bootstrap()
    chat = await conversation.create_chat(identity.owner_id, identity.character_id, "chat-1")
    other = await conversation.create_chat(identity.owner_id, identity.character_id, "chat-2")
    await conversation.accept(identity.owner_id, chat.id, "request", "原始输入")
    for target, content in [(chat.id, "变更输入"), (other.id, "原始输入")]:
        with pytest.raises(DomainError, match="idempotency_conflict"):
            await conversation.accept(identity.owner_id, target, "request", content)


async def test_concurrent_tabs_share_one_active_turn_across_chats(conversation):
    import asyncio

    from hmph.contracts import DomainError

    identity = await conversation.bootstrap()
    chats = [
        await conversation.create_chat(identity.owner_id, identity.character_id, f"chat-{i}")
        for i in range(2)
    ]
    results = await asyncio.gather(
        *[
            conversation.accept(identity.owner_id, chat.id, f"request-{i}", "你好")
            for i, chat in enumerate(chats)
        ],
        return_exceptions=True,
    )
    assert sum(not isinstance(result, Exception) for result in results) == 1
    errors = [result.code for result in results if isinstance(result, DomainError)]
    assert errors == ["turn_in_progress"]


async def test_another_owner_cannot_read_cancel_or_write_the_turn(conversation):
    from uuid import uuid4

    import pytest

    from hmph.contracts import DomainError

    identity = await conversation.bootstrap()
    outsider = await conversation.bootstrap(uuid4())
    chat = await conversation.create_chat(identity.owner_id, identity.character_id, "chat")
    turn = await conversation.accept(identity.owner_id, chat.id, "turn", "私密输入")
    for operation in (
        lambda: conversation.get_turn(outsider.owner_id, turn.id),
        lambda: conversation.events(outsider.owner_id, turn.id),
        lambda: conversation.messages(outsider.owner_id, chat.id),
        lambda: conversation.cancel(outsider.owner_id, turn.id, "cancel"),
        lambda: conversation.accept(outsider.owner_id, chat.id, "turn", "越权输入"),
    ):
        with pytest.raises(DomainError, match="not_found"):
            await operation()
