import pytest

from hmph.contracts import DomainError


@pytest.mark.parametrize("still_open", [False, True])
async def test_forget_does_not_reactivate_an_answered_question(conversation, still_open):
    from hmph.dialogue import Dialogue
    from hmph.dialogue_types import Anchor, StatePatch
    from hmph.memory import Memory

    identity = await conversation.bootstrap()
    owner = identity.owner_id
    chat = await conversation.create_chat(owner, identity.character_id, "chat")

    async def complete(request_id, user, reply):
        await conversation.accept(owner, chat.id, request_id, user)
        lease = await conversation.claim("generator")
        await conversation.generation_request(lease)
        await conversation.publish(lease, 0, reply)
        await conversation.finish(lease)
        return (await conversation.messages(owner, chat.id))[-2]

    await complete("practice", "今天练了琴。", "练得挺认真。")
    poem = await complete("poem", "那首诗叫山居秋暝。", "知道了。")
    old = await complete("old", "我们现在还有什么共同计划？", "每周聊一次练琴。")
    goal = Anchor(message_id=old.id, text=old.content, quote=old.content)
    dialogue = Dialogue(conversation.db)
    await dialogue.commit(
        await dialogue.claim("state"),
        StatePatch(topic=goal, goal=goal, open_items=[goal] if still_open else []),
    )

    memory = Memory(conversation.db, digest_key=conversation.digest_key)
    receipt = await memory.forget_source(owner, poem.id, "forget", expected_revision=1)
    assert receipt.turns == 2
    assert (await dialogue.read(owner, chat.id)).data.goal.message_id == old.id
    await conversation.accept(owner, chat.id, "current", "还记得那首诗的题目吗？")
    request = await conversation.generation_request(await conversation.claim("current"))

    assert [message.content for message in request.messages] == [
        "今天练了琴。",
        "练得挺认真。",
        old.content,
        "还记得那首诗的题目吗？",
    ]
    assert (request.short_term.data.goal is not None) == still_open
    assert (request.short_term.data.topic is not None) == still_open
    assert "山居秋暝" not in request.model_dump_json()


async def source_turn(conversation, identity, chat, request_id, content):
    turn = await conversation.accept(identity.owner_id, chat.id, request_id, content)
    await conversation.cancel(identity.owner_id, turn.id, request_id)
    return (await conversation.messages(identity.owner_id, chat.id))[-1]


async def test_dialogue_state_keeps_grounded_goal_and_contiguous_watermark(conversation):
    from hmph.dialogue import Dialogue
    from hmph.dialogue_types import Anchor, StatePatch

    identity = await conversation.bootstrap()
    chat = await conversation.create_chat(identity.owner_id, identity.character_id, "chat")
    message = await source_turn(
        conversation, identity, chat, "plan", "周五去星河公司面试，先听我吐槽。"
    )
    dialogue = Dialogue(conversation.db)
    lease = await dialogue.claim("state-worker")
    assert lease.through_seq == 2
    goal = Anchor(message_id=message.id, text="周五去星河公司面试", quote="周五去星河公司面试")
    result = await dialogue.commit(lease, StatePatch(goal=goal, open_items=[goal]))
    assert result.status == "committed"
    view = await dialogue.read(identity.owner_id, chat.id)
    assert view.processed_through_seq == 2 and view.pending_messages == 0
    assert view.data.goal.text == "周五去星河公司面试"
    assert view.data.open_items[0].status == "open"
    assert view.data.goal.message_revision == 1


async def test_unprocessed_correction_remains_visible_and_old_commit_cannot_advance_watermark(
    conversation,
):
    from hmph.dialogue import Dialogue
    from hmph.dialogue_types import Anchor, StatePatch

    identity = await conversation.bootstrap()
    chat = await conversation.create_chat(identity.owner_id, identity.character_id, "chat")
    message = await source_turn(conversation, identity, chat, "old", "周五去星河公司面试。")
    dialogue = Dialogue(conversation.db)
    first = await dialogue.claim("state-worker")
    goal = Anchor(message_id=message.id, text="周五去星河公司面试", quote="周五去星河公司面试")
    await dialogue.commit(first, StatePatch(goal=goal, open_items=[goal]))
    await conversation.accept(
        identity.owner_id, chat.id, "correction", "不去星河公司了，改去远山公司。"
    )
    request = await conversation.generation_request(await conversation.claim("generator"))
    assert request.messages[-1].content == "不去星河公司了，改去远山公司。"
    assert request.short_term.processed_through_seq == 2
    assert request.short_term.omitted_delta_messages == 0 and not request.short_term.pending_gap
    assert request.short_term.data.goal.text == "周五去星河公司面试"
    assert (await dialogue.commit(first, StatePatch(goal=goal))).status == "stale"
    assert (await dialogue.read(identity.owner_id, chat.id)).processed_through_seq == 2


async def test_forgetting_purges_state_and_its_derived_reply_and_rejects_old_state_worker(
    conversation,
):
    from hmph.dialogue import Dialogue
    from hmph.dialogue_types import Anchor, StatePatch
    from hmph.memory import Memory
    from hmph.memory_types import MemoryProposal

    identity = await conversation.bootstrap()
    chat = await conversation.create_chat(identity.owner_id, identity.character_id, "chat")
    message = await source_turn(conversation, identity, chat, "fact", "我叫小沐，周五去面试。")
    memory = Memory(conversation.db, digest_key=conversation.digest_key)
    await memory.learn(
        await memory.claim("extractor"),
        [
            MemoryProposal(predicate="name", value="小沐", quote="我叫小沐"),
        ],
    )
    fact = (await memory.entries(identity.owner_id, identity.character_id))[0]
    dialogue = Dialogue(conversation.db)
    await dialogue.commit(
        await dialogue.claim("state-worker"),
        StatePatch(goal=Anchor(message_id=message.id, text="周五去面试", quote="周五去面试")),
    )
    derived = await conversation.accept(identity.owner_id, chat.id, "question", "我有什么安排？")
    generator = await conversation.claim("generator")
    request = await conversation.generation_request(generator)
    assert request.short_term.data.goal.text == "周五去面试"
    await conversation.publish(generator, 0, "周五面试。")
    await conversation.finish(generator)
    old_worker = await dialogue.claim("slow-state-worker")

    assert (await memory.preview_forget(identity.owner_id, fact.id)).dialogue_states == 1
    receipt = await memory.forget(identity.owner_id, fact.id, "forget", expected_revision=1)
    assert receipt.dialogue_states == 1
    state = await dialogue.read(identity.owner_id, chat.id)
    assert state.data.goal is None
    assert (await dialogue.commit(old_worker, StatePatch())).status == "stale"
    assert "周五面试" not in str(await conversation.events(identity.owner_id, derived.id))


async def test_explicit_drop_closes_open_item_and_clears_matching_goal(conversation):
    from hmph.dialogue import Dialogue
    from hmph.dialogue_types import Anchor, StatePatch, Transition

    identity = await conversation.bootstrap()
    chat = await conversation.create_chat(identity.owner_id, identity.character_id, "chat")
    message = await source_turn(conversation, identity, chat, "plan", "周五去面试。")
    dialogue = Dialogue(conversation.db)
    goal = Anchor(message_id=message.id, text="周五去面试", quote="周五去面试")
    await dialogue.commit(
        await dialogue.claim("state-worker"), StatePatch(goal=goal, open_items=[goal])
    )
    item = (await dialogue.read(identity.owner_id, chat.id)).data.open_items[0]
    correction = await source_turn(conversation, identity, chat, "drop", "取消周五的面试安排。")
    await dialogue.commit(
        await dialogue.claim("state-worker"),
        StatePatch(
            transitions=[
                Transition(
                    item_id=item.id,
                    status="dropped",
                    evidence=Anchor(
                        message_id=correction.id,
                        text="取消周五的面试安排",
                        quote="取消周五的面试安排",
                    ),
                )
            ]
        ),
    )
    view = await dialogue.read(identity.owner_id, chat.id)
    assert view.data.goal is None and view.data.open_items[0].status == "dropped"
    assert view.processed_through_seq == 4


async def test_old_state_worker_cannot_commit_after_expiration_and_reclaim(conversation):
    import asyncio

    from hmph.dialogue import Dialogue
    from hmph.dialogue_types import StatePatch

    identity = await conversation.bootstrap()
    chat = await conversation.create_chat(identity.owner_id, identity.character_id, "chat")
    await source_turn(conversation, identity, chat, "plan", "周五去面试。")
    dialogue = Dialogue(conversation.db)
    first = await dialogue.claim("lost", lease_seconds=0.01)
    await asyncio.sleep(0.02)
    assert await dialogue.recover_expired() == 1
    second = await dialogue.claim("replacement")
    assert (await dialogue.commit(first, StatePatch())).status == "stale"
    assert (await dialogue.commit(second, StatePatch())).status == "committed"


async def test_backlog_larger_than_recent_window_marks_gap_and_does_not_claim_complete_state(
    conversation,
):
    identity = await conversation.bootstrap()
    chat = await conversation.create_chat(identity.owner_id, identity.character_id, "chat")
    for index in range(44):
        await source_turn(conversation, identity, chat, f"m-{index}", f"近期输入 {index}")
    await conversation.accept(identity.owner_id, chat.id, "latest", "取消之前计划，改成周日。")
    request = await conversation.generation_request(await conversation.claim("generator"))
    assert request.messages[-1].content == "取消之前计划，改成周日。"
    assert request.short_term.processed_through_seq == 0
    assert request.short_term.pending_gap and request.short_term.omitted_delta_messages == 5


async def test_state_rejects_unsupported_quotes_and_isolates_other_owner(conversation):
    from uuid import uuid4

    from hmph.dialogue import Dialogue
    from hmph.dialogue_types import Anchor, StatePatch

    identity = await conversation.bootstrap()
    outsider = await conversation.bootstrap(uuid4())
    chat = await conversation.create_chat(identity.owner_id, identity.character_id, "chat")
    message = await source_turn(conversation, identity, chat, "plan", "周五去面试。")
    dialogue = Dialogue(conversation.db)
    lease = await dialogue.claim("state-worker")
    with pytest.raises(DomainError, match="unsupported_state"):
        await dialogue.commit(
            lease, StatePatch(goal=Anchor(message_id=message.id, text="长期焦虑", quote="长期焦虑"))
        )
    assert (await dialogue.read(identity.owner_id, chat.id)).processed_through_seq == 0
    with pytest.raises(DomainError, match="not_found"):
        await dialogue.read(outsider.owner_id, chat.id)


async def test_condensed_model_label_is_replaced_by_its_verified_original_quote(conversation):
    from hmph.dialogue import Dialogue
    from hmph.dialogue_types import Anchor, StatePatch

    identity = await conversation.bootstrap()
    chat = await conversation.create_chat(identity.owner_id, identity.character_id, "chat")
    message = await source_turn(conversation, identity, chat, "plan", "周五我要去星河公司面试。")
    dialogue = Dialogue(conversation.db)
    await dialogue.commit(
        await dialogue.claim("worker"),
        StatePatch(
            open_items=[
                Anchor(message_id=message.id, text="周五去面试", quote="周五我要去星河公司面试")
            ]
        ),
    )
    view = await dialogue.read(identity.owner_id, chat.id)
    assert view.data.open_items[0].content.text == "周五我要去星河公司面试"


async def test_rejected_state_does_not_advance_watermark_and_retains_known_provider_cost(
    conversation,
):
    from hmph.dialogue import Dialogue
    from hmph.dialogue_types import Anchor, StatePatch
    from hmph.dialogue_worker import StateWorker
    from hmph.model_gateway import ModelUsage

    identity = await conversation.bootstrap()
    chat = await conversation.create_chat(identity.owner_id, identity.character_id, "chat")
    source = await source_turn(conversation, identity, chat, "plan", "周五去面试。")

    async def invalid_proposal(lease):
        return StatePatch(
            goal=Anchor(message_id=source.id, text="虚构事实", quote="虚构事实")
        ), ModelUsage(prompt_tokens=100, completion_tokens=40, total_tokens=140)

    dialogue = Dialogue(conversation.db)
    await StateWorker(dialogue, invalid_proposal).run_once()
    view = await dialogue.read(identity.owner_id, chat.id)
    assert view.task_status == "failed" and view.processed_through_seq == 0
    assert view.usage["total_tokens"] == 140
