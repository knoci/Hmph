import pytest
from sqlalchemy import select

from hmph.conversation import Conversation
from hmph.dialogue import Dialogue
from hmph.dialogue_types import Anchor, StatePatch
from hmph.memory_builds import MemoryBuilds
from hmph.models import Message, Turn
from hmph.online_memory import MemorySelection, NativeSources
from hmph.working_memory import contextual_query, retrieve


async def grounded_goal(conversation, *, nested_topic=False, state_pending=False):
    identity = await conversation.bootstrap()
    chat = await conversation.create_chat(identity.owner_id, identity.character_id, "context-goal")
    body = "我想接着聊远山公司面试。" if state_pending else "周五去远山公司面试。"
    first = await conversation.accept(identity.owner_id, chat.id, "goal", body)
    lease = await conversation.claim("context-goal")
    await conversation.finish(lease)
    message = next(
        m for m in await conversation.messages(identity.owner_id, chat.id) if m.turn_id == first.id
    )
    if state_pending:
        return identity, chat, message
    dialogue = Dialogue(conversation.db)
    state_lease = await dialogue.claim("context-state")
    await dialogue.commit(
        state_lease,
        StatePatch(
            goal=Anchor(message_id=message.id, text="远山公司面试", quote="周五去远山公司面试"),
            topic=Anchor(message_id=message.id, text="远山公司", quote="远山公司")
            if nested_topic
            else None,
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


@pytest.mark.parametrize("state_pending", [False, True])
async def test_timem_follows_grounded_reference_to_another_chat(conversation, state_pending):
    identity, chat, anchor = await grounded_goal(
        conversation, nested_topic=True, state_pending=state_pending
    )
    owner, character = identity.owner_id, identity.character_id
    last_chat = await conversation.create_chat(owner, character, "earlier-interview")
    await conversation.accept(owner, last_chat.id, "result", "远山公司的面试改到周六下午。")
    await conversation.finish(await conversation.claim("result"))
    result = (await conversation.messages(owner, last_chat.id))[0]
    builds = MemoryBuilds(conversation.db, digest_key=conversation.digest_key)
    build = await builds.request(owner, character, "index", backend="mem0_timem")
    writer = await builds.claim("index", backend="mem0_timem")
    await builds.quiesce(writer)
    assert await builds.publish(writer)
    await MemorySelection(
        conversation.db, digest_key=conversation.digest_key, allow_timem=True
    ).update(owner, character, "select", build_id=build.id, expected_revision=1)

    queries = []

    class Reader:
        async def retrieve_sources(self, ticket, query):
            queries.append(query)
            ids = (result.id,) if "远山公司面试" in query else ()
            return NativeSources(source_ids=ids)

    service = Conversation(
        conversation.db, digest_key=conversation.digest_key, source_reader=Reader()
    )
    question = "那家后来怎样了？"
    await service.accept(owner, chat.id, "follow-up", question)
    lease = await service.claim("follow-up")
    request = await service.generation_request(lease)
    expected_anchor = anchor.content if state_pending else "周五去远山公司面试"
    assert queries == [question + " " + expected_anchor]
    assert request.messages[-1].content == question
    assert request.workspace.query_hint_source_ids == (anchor.id,)
    assert request.workspace.backend == "mem0_timem"
    assert [(s.message_id, s.content) for s in request.raw_evidence] == [
        (result.id, "远山公司的面试改到周六下午。")
    ]
    await service.finish(lease)

    correction = "不是那家，改成北川了。"
    await service.accept(owner, chat.id, "correct-interview", correction)
    lease = await service.claim("correction")
    request = await service.generation_request(lease)
    assert queries[-1] == correction
    assert request.workspace.query_hint_source_ids == ()
    await service.finish(lease)
