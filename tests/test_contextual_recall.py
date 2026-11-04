from sqlalchemy import select

from hmph.dialogue import Dialogue
from hmph.dialogue_types import Anchor, StatePatch
from hmph.models import Message, Turn
from hmph.working_memory import contextual_query, retrieve


async def grounded_goal(conversation):
    identity = await conversation.bootstrap()
    chat = await conversation.create_chat(identity.owner_id, identity.character_id, "context-goal")
    first = await conversation.accept(identity.owner_id, chat.id, "goal", "周五去远山公司面试。")
    lease = await conversation.claim("context-goal")
    await conversation.finish(lease)
    message = next(
        m for m in await conversation.messages(identity.owner_id, chat.id) if m.turn_id == first.id
    )
    dialogue = Dialogue(conversation.db)
    state_lease = await dialogue.claim("context-state")
    await dialogue.commit(
        state_lease,
        StatePatch(
            goal=Anchor(message_id=message.id, text="远山公司面试", quote="周五去远山公司面试")
        ),
    )
    return identity, chat, message


async def test_grounded_deictic_query_finds_original_and_records_actual_hint(conversation):
    identity, chat, source = await grounded_goal(conversation)
    query = "那家后来怎样了？"
    turn = await conversation.accept(identity.owner_id, chat.id, "deictic", query)
    lease = await conversation.claim("context-query")
    request = await conversation.generation_request(lease)
    assert request.workspace.query_hint_source_ids == (source.id,)
    assert source.id in {s.message_id for s in request.raw_evidence}
    assert request.companion.act == "recall"
    audit = await conversation.memory_explanation(identity.owner_id, turn.id)
    assert audit["context"]["workspace"]["state_hint_count"] == 1
    async with conversation.db.sessions() as session:
        row = await session.get(Turn, turn.id)
        # A deliberately lagging-state control uses the same new scenario, not
        # a repeated historical suite. It cannot inject an old query hint.
        lag = request.short_term.model_copy(update={"processed_through_seq": 0})
        baseline = await retrieve(session, row, query, lag)
        assert baseline.query_hint_source_ids == ()
        assert source.id not in {s.message_id for s in baseline.raw}
    await conversation.finish(lease)


async def test_correction_boundary_and_source_revision_disable_context_expansion(conversation):
    identity, chat, source = await grounded_goal(conversation)
    turn = await conversation.accept(identity.owner_id, chat.id, "safety", "那家后来怎样了？")
    lease = await conversation.claim("context-safety")
    request = await conversation.generation_request(lease)
    async with conversation.db.sessions.begin() as session:
        row = await session.get(Turn, turn.id)
        state = request.short_term
        original = "那家不是远山公司，改成别家。"
        assert await contextual_query(session, row, original, state, {}) == (original, ())
        query = "那家后来怎样了？"
        preferences = {"_boundaries": [{"target": "远山公司", "kind": "topic", "mode": "never"}]}
        assert await contextual_query(session, row, query, state, preferences) == (query, ())
        message = await session.scalar(select(Message).where(Message.id == source.id))
        message.revision += 1
        await session.flush()
        assert await contextual_query(session, row, query, state, {}) == (query, ())
