import pytest

from hmph.dialogue import Dialogue
from hmph.dialogue_types import Anchor, StatePatch


@pytest.mark.parametrize("lag", [0, 1, 5])
async def test_grounded_state_and_full_delta_remain_usable_with_lag(conversation, lag):
    identity = await conversation.bootstrap()
    chat = await conversation.create_chat(identity.owner_id, identity.character_id, "lag")

    async def say(content, key):
        turn = await conversation.accept(identity.owner_id, chat.id, key, content)
        lease = await conversation.claim("lag-fixture")
        await conversation.finish(lease)
        return next(
            m
            for m in await conversation.messages(identity.owner_id, chat.id)
            if m.turn_id == turn.id
        )

    goal = await say("我周六要去北岸工作室面试。", "goal")
    dialogue = Dialogue(conversation.db)
    state_lease = await dialogue.claim("lag-state")
    await dialogue.commit(
        state_lease,
        StatePatch(
            goal=Anchor(
                message_id=goal.id,
                text="北岸工作室面试",
                quote="我周六要去北岸工作室面试。",
            )
        ),
    )
    for index in range(lag):
        await say(f"刚整理好第{index + 1}张书桌。", f"lag-{index}")
    turn = await conversation.accept(identity.owner_id, chat.id, "query", "那家面试有什么安排？")
    lease = await conversation.claim("lag-query")
    calls = []

    async def planner(request):
        calls.append(request)
        raise AssertionError("Supported goal slots need no extra model")

    conversation.recall_planner = planner
    request = await conversation.generation_request(lease)
    assert not calls
    assert request.short_term.data.goal.message_id == goal.id
    assert not request.short_term.pending_gap
    assert goal.id in {s.message_id for s in request.raw_evidence}
    assert bool(request.workspace.query_hint_source_ids) is (lag == 0)
    assert request.messages[-1].content == "那家面试有什么安排？"
    explanation = await conversation.memory_explanation(identity.owner_id, turn.id)
    assert explanation["context"]["planner"]["attempts"] == 0
