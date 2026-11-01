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
