async def test_missing_structured_fact_triggers_bounded_original_source_lookup(conversation):
    identity = await conversation.bootstrap()
    old = await conversation.create_chat(identity.owner_id, identity.character_id, "old")
    turn = await conversation.accept(
        identity.owner_id, old.id, "helmet", "我上次买的头盔是 K2 Horizon。"
    )
    await conversation.cancel(identity.owner_id, turn.id, "cancel")
    source = (await conversation.messages(identity.owner_id, old.id))[0]
    new = await conversation.create_chat(identity.owner_id, identity.character_id, "new")
    await conversation.accept(identity.owner_id, new.id, "question", "我上次买的头盔是什么？")
    request = await conversation.generation_request(await conversation.claim("generator"))
    assert request.workspace.query_count == 2
    assert request.workspace.rounds == 2
    assert request.raw_evidence[0].message_id == source.id
    assert "K2 Horizon" in request.raw_evidence[0].content
    assert request.workspace.slots[0].status == "unknown"
    assert request.workspace.slots[0].candidate_source_ids == (source.id,)
    assert request.workspace.coverage == "partial"


async def test_memory_only_source_is_registered_for_mention_and_forgetting(conversation):
    from hmph.memory import Memory
    from hmph.memory_types import MemoryProposal
    from hmph.model_gateway import SourceMention
    from hmph.models import TurnSourceUse

    identity = await conversation.bootstrap()
    old = await conversation.create_chat(identity.owner_id, identity.character_id, "old")
    turn = await conversation.accept(
        identity.owner_id, old.id, "fact", "I currently work at River Glass."
    )
    await conversation.cancel(identity.owner_id, turn.id, "cancel")
    source = (await conversation.messages(identity.owner_id, old.id))[0]
    memory = Memory(conversation.db, digest_key=conversation.digest_key)
    await memory.learn(
        await memory.claim("extract"),
        [
            MemoryProposal(
                predicate="employer", value="River Glass", quote="I currently work at River Glass."
            )
        ],
    )
    chat = await conversation.create_chat(identity.owner_id, identity.character_id, "new")
    query = await conversation.accept(
        identity.owner_id, chat.id, "query", "Where do I currently work?"
    )
    lease = await conversation.claim("generation")
    req = await conversation.generation_request(lease)
    assert req.evidence and not req.raw_evidence
    assert source.id not in {o.message_id for o in req.history_origins}
    async with conversation.db.sessions() as s:
        use = await s.get(TurnSourceUse, (query.id, source.id))
        assert use is not None and use.message_revision == source.revision
    hint = SourceMention(source_id=source.id, quote="River Glass")
    assert await conversation.check_companion_output(lease, "你现在在 River Glass 工作。", (hint,))
    await conversation.publish(lease, 0, "你现在在 River Glass 工作。", mentions=(hint,))
    await conversation.finish(lease)
    fact = req.evidence[0]
    receipt = await memory.forget(
        identity.owner_id, fact.id, "forget", expected_revision=fact.revision
    )
    assert receipt.status == "completed"
    assert all(
        m.deleted and "River Glass" not in m.content
        for m in await conversation.messages(identity.owner_id, chat.id)
        if m.role == "assistant"
    )


async def test_raw_lookup_preserves_source_local_calendar_and_timezone(conversation):
    from datetime import UTC, datetime

    from sqlalchemy import update

    from hmph.models import Turn

    identity = await conversation.bootstrap()
    chat = await conversation.create_chat(identity.owner_id, identity.character_id, "source")
    source = await conversation.accept(identity.owner_id, chat.id, "fact", "我昨天参加了合唱排练。")
    await conversation.cancel(identity.owner_id, source.id, "cancel")
    async with conversation.db.sessions.begin() as session:
        await session.execute(
            update(Turn)
            .where(Turn.id == source.id)
            .values(
                created_at=datetime(2026, 9, 5, 6, 30, tzinfo=UTC),
                source_timezone="America/Los_Angeles",
            )
        )
    current = await conversation.create_chat(identity.owner_id, identity.character_id, "new")
    await conversation.accept(identity.owner_id, current.id, "query", "我上次合唱排练是什么时候？")
    req = await conversation.generation_request(await conversation.claim("reader"))
    raw = next(s for s in req.raw_evidence if s.content == "我昨天参加了合唱排练。")
    assert raw.source_timezone == "America/Los_Angeles"
    assert raw.recorded_at.isoformat() == "2026-09-04T23:30:00-07:00"


async def test_common_english_words_cannot_displace_the_old_matching_original(conversation):
    identity = await conversation.bootstrap()
    old = await conversation.create_chat(identity.owner_id, identity.character_id, "old")
    fact = await conversation.accept(
        identity.owner_id, old.id, "fact", "I received a drum recently."
    )
    await conversation.cancel(identity.owner_id, fact.id, "cancel")
    original = (await conversation.messages(identity.owner_id, old.id))[0]
    for i in range(16):
        distractor = await conversation.accept(
            identity.owner_id, old.id, f"distractor-{i}", f"The parcel was the item number {i}."
        )
        await conversation.cancel(identity.owner_id, distractor.id, f"cancel-{i}")
    new = await conversation.create_chat(identity.owner_id, identity.character_id, "new")
    await conversation.accept(
        identity.owner_id, new.id, "query", "On what day was the drum delivered?"
    )
    req = await conversation.generation_request(await conversation.claim("reader"))
    assert original.id in {s.message_id for s in req.raw_evidence}
    assert req.workspace.candidate_count <= 80


async def test_forget_retracts_reply_derived_only_from_raw_lookup(conversation):
    from hmph.memory import Memory
    from hmph.memory_types import MemoryProposal

    identity = await conversation.bootstrap()
    old = await conversation.create_chat(identity.owner_id, identity.character_id, "old")
    turn = await conversation.accept(
        identity.owner_id, old.id, "fact", "我叫小沐。我买的头盔是 K2 Horizon。"
    )
    await conversation.cancel(identity.owner_id, turn.id, "cancel")
    memory = Memory(conversation.db, digest_key=conversation.digest_key)
    await memory.learn(
        await memory.claim("extractor"),
        [MemoryProposal(predicate="name", value="小沐", quote="我叫小沐")],
    )
    fact = (await memory.entries(identity.owner_id, identity.character_id))[0]
    new = await conversation.create_chat(identity.owner_id, identity.character_id, "new")
    question = await conversation.accept(
        identity.owner_id, new.id, "question", "我买的头盔是什么？"
    )
    lease = await conversation.claim("generator")
    request = await conversation.generation_request(lease)
    assert request.raw_evidence and not request.evidence
    await conversation.publish(lease, 0, "K2 Horizon。")
    await conversation.finish(lease)
    await memory.forget(identity.owner_id, fact.id, "forget", expected_revision=1)
    assert "K2 Horizon" not in str(await conversation.events(identity.owner_id, question.id))


async def test_raw_excerpt_finds_dense_fact_after_generic_prefix_and_fits_chinese_budget(
    conversation,
):
    identity = await conversation.bootstrap()
    old = await conversation.create_chat(identity.owner_id, identity.character_id, "old")
    text = (
        "提到头盔只是无关介绍。"
        + "这是一段无关的背景说明。" * 80
        + "我购买的滑雪头盔型号是 K2 Horizon，后来已经归还给店家。"
        + "其他无关说明。" * 50
    )
    turn = await conversation.accept(identity.owner_id, old.id, "source", text)
    await conversation.cancel(identity.owner_id, turn.id, "cancel")
    new = await conversation.create_chat(identity.owner_id, identity.character_id, "new")
    await conversation.accept(
        identity.owner_id, new.id, "question", "我购买的滑雪头盔型号是什么，后来如何处理？"
    )
    request = await conversation.generation_request(await conversation.claim("generator"))
    assert request.raw_evidence and "K2 Horizon" in request.raw_evidence[0].content
    assert "已经归还" in request.raw_evidence[0].content
    source = request.raw_evidence[0]
    assert text[source.start_offset : source.end_offset] == source.content
