"""Companion behavior through the existing turn, settings and forget interfaces."""

import pytest

from hmph.preferences import Preferences


def test_personal_history_enumeration_allows_requested_sources_instead_of_proactive_cap():
    from uuid import uuid4

    from hmph.companion_policy import SourceCandidate, decide

    sources = [
        SourceCandidate(uuid4(), text) for text in ("我去过杭州。", "我去过苏州。", "我去过南京。")
    ]
    plan = decide("我一共去过哪些地方？", {}, sources, policy_revision=1)
    assert all(d.permission == "mention" and d.reason == "requested" for d in plan.memory_use)


async def test_longer_replacement_nickname_is_allowed_but_former_standalone_name_is_blocked(
    conversation,
):
    identity = await conversation.bootstrap()
    chat, _, lease, _ = await request(conversation, identity, "以后叫我阿禾。", "first")
    await conversation.finish(lease)
    _, _, lease, context = await request(
        conversation, identity, "以后叫我阿禾同学。", "longer", chat=chat
    )
    assert context.preferences.nickname == "阿禾同学"
    assert await conversation.check_companion_output(lease, "阿禾同学，收到。")
    assert not await conversation.check_companion_output(lease, "阿禾，收到。")
    await conversation.finish(lease)


async def test_topic_boundary_also_masks_conflicting_expression_nickname(conversation):
    identity = await conversation.bootstrap()
    chat, _, lease, _ = await request(conversation, identity, "以后叫我小木。", "name")
    await conversation.finish(lease)
    _, _, lease, context = await request(
        conversation, identity, "以后不要提小木。", "topic", chat=chat
    )
    assert "小木" not in context.model_dump_json()
    assert context.preferences.nickname is None
    await conversation.finish(lease)


async def request(conversation, identity, text, key, *, chat=None):
    chat = chat or await conversation.create_chat(identity.owner_id, identity.character_id, key)
    turn = await conversation.accept(identity.owner_id, chat.id, key, text)
    lease = await conversation.claim("companion-test")
    return chat, turn, lease, await conversation.generation_request(lease)


async def test_explicit_nickname_repair_applies_to_own_turn_through_twenty_turns_and_new_chat(
    conversation,
):
    identity = await conversation.bootstrap()
    chat, turn, lease, context = await request(conversation, identity, "以后叫我小禾。", "fix")
    assert context.preferences.nickname == "小禾"
    assert context.companion.act == "repair"
    assert context.companion.repair_action == "applied"
    assert lease.policy_revision == 2
    await conversation.publish(lease, 0, "好，小禾。")
    await conversation.finish(lease)
    for index in range(1, 21):
        _, _, lease, context = await request(
            conversation, identity, "你好。", f"after-{index}", chat=chat if index < 20 else None
        )
        assert context.preferences.nickname == "小禾"
        await conversation.finish(lease)
    settings = await Preferences(conversation).read(identity.owner_id, identity.character_id)
    assert settings["revision"] == 2
    audit = await conversation.companion_explanation(identity.owner_id, turn.id)
    assert audit["repairs"][0]["status"] == "corrected"
    assert audit["repairs"][0]["kind"] == "nickname"


@pytest.mark.parametrize(
    "text,act,mode",
    [
        ("先听我说，我只想吐槽，不需要建议。", "acknowledge", "listen"),
        ("好消息，我拿到了offer！", "celebrate", "unknown"),
        ("你又叫错了。", "clarify", "unknown"),
        ("先这样，晚安。", "close", "unknown"),
    ],
)
async def test_explicit_current_intent_selects_companion_act(conversation, text, act, mode):
    identity = await conversation.bootstrap()
    _, _, lease, context = await request(conversation, identity, text, "intent")
    assert context.companion.act == act and context.companion.support_mode == mode
    if act in {"acknowledge", "clarify"}:
        assert context.companion.operator == "direct_care"
    await conversation.finish(lease)


async def test_asking_about_the_word_goodnight_does_not_close_the_conversation(conversation):
    identity = await conversation.bootstrap()
    _, _, lease, context = await request(conversation, identity, "晚安用英语怎么说？", "word")
    assert context.companion.act != "close"
    await conversation.finish(lease)


async def test_short_term_listening_signal_survives_without_resending_sensitive_story(conversation):
    from hmph.dialogue import Dialogue
    from hmph.dialogue_types import Anchor, ModeCue, StatePatch

    identity = await conversation.bootstrap()
    chat, _, lease, _ = await request(
        conversation, identity, "先听我说那次失恋，我只是想吐槽。", "story"
    )
    await conversation.finish(lease)
    dialogue = Dialogue(conversation.db)
    state = await dialogue.claim("state")
    source = state.messages[0]
    await dialogue.commit(
        state,
        StatePatch(
            response_mode=ModeCue(
                mode="listen",
                evidence=Anchor(message_id=source.id, text="先听我说", quote="先听我说"),
            )
        ),
    )
    _, turn, lease, context = await request(
        conversation, identity, "然后我就沉默了。", "continue", chat=chat
    )
    assert context.companion.support_mode == "listen"
    assert context.companion.signal_source == source.id
    assert "失恋" not in context.model_dump_json()
    assert (
        source.id
        in (await conversation.memory_explanation(identity.owner_id, turn.id))["source_ids"]
    )
    await conversation.finish(lease)


async def test_current_nickname_correction_blocks_old_generation_before_review(conversation):
    import json

    import httpx
    from pydantic import SecretStr

    from hmph.deepseek import DeepSeekModel
    from hmph.memory import Memory
    from hmph.memory_types import MemoryProposal
    from hmph.model_gateway import FALLBACK
    from hmph.worker import Worker

    identity = await conversation.bootstrap()
    chat, _, lease, _ = await request(conversation, identity, "我叫小沐。", "old")
    await conversation.finish(lease)
    memory = Memory(conversation.db, digest_key=conversation.digest_key)
    await memory.learn(
        await memory.claim("learn"),
        [MemoryProposal(predicate="name", value="小沐", quote="我叫小沐")],
    )
    turn = await conversation.accept(
        identity.owner_id, chat.id, "new", "以后叫我小禾。", model_mode="deepseek"
    )
    seen = []

    def endpoint(request):
        body = json.loads(request.content)
        seen.append(body)
        assert "小沐" not in json.dumps(body, ensure_ascii=False)
        assert body["stream"]  # Known forbidden draft must not enter another model call.
        return httpx.Response(
            200,
            text="data: "
            + json.dumps({"choices": [{"delta": {"content": "小沐，知道啦。"}}]})
            + "\n\n"
            + "data: "
            + json.dumps({"choices": [{"delta": {}, "finish_reason": "stop"}]})
            + "\n\n"
            + "data: [DONE]\n\n",
        )

    model = DeepSeekModel(
        SecretStr("companion-test-no-real-key"), transport=httpx.MockTransport(endpoint)
    )
    await Worker(conversation, model, model_mode="deepseek", reviewer=model.review).run_once()
    assert len(seen) == 1
    assert (await conversation.messages(identity.owner_id, chat.id))[-1].content == FALLBACK
    saved = await conversation.get_turn(identity.owner_id, turn.id)
    assert saved.status == "completed" and saved.usage["guard_action"] == "policy_fallback"


async def test_graph_source_on_request_boundary_uses_only_allowed_contiguous_original(conversation):
    from test_online_memory import NativeReader, ready_graph

    from hmph.conversation import Conversation

    identity = await conversation.bootstrap()
    chat, _, lease, _ = await request(
        conversation, identity, "那次失恋让我难过，我喜欢跑步。", "story"
    )
    await conversation.finish(lease)
    build, _, ids = await ready_graph(conversation, identity)
    from hmph.online_memory import MemorySelection

    await MemorySelection(conversation.db, digest_key=conversation.digest_key).update(
        identity.owner_id, identity.character_id, "select", expected_revision=1, build_id=build.id
    )
    _, _, lease, _ = await request(
        conversation, identity, "以后别主动提我的失恋。", "boundary", chat=chat
    )
    await conversation.finish(lease)
    service = Conversation(
        conversation.db, digest_key=conversation.digest_key, source_reader=NativeReader(ids)
    )
    _, _, lease, context = await request(
        service, identity, "你记得我喜欢的跑步吗？", "safe", chat=chat
    )
    assert "失恋" not in context.model_dump_json()
    assert any(s.content == "我喜欢跑步" for s in context.raw_evidence)
    await service.finish(lease)
    _, _, lease, context = await request(service, identity, "你记得我那次失恋吗？", "allowed")
    assert any("失恋" in s.content for s in context.raw_evidence)
    await service.finish(lease)


async def test_topic_boundary_removes_old_text_from_history_raw_memory_and_state(conversation):
    from hmph.dialogue import Dialogue
    from hmph.dialogue_types import Anchor, StatePatch
    from hmph.memory import Memory
    from hmph.memory_types import MemoryProposal

    identity = await conversation.bootstrap()
    chat, _, lease, _ = await request(
        conversation, identity, "那次失恋让我难过，我现在喜欢跑步。", "story"
    )
    await conversation.finish(lease)
    memory = Memory(conversation.db, digest_key=conversation.digest_key)
    await memory.learn(
        await memory.claim("learn"),
        [MemoryProposal(predicate="experience", value="失恋", quote="那次失恋让我难过")],
    )
    state = Dialogue(conversation.db)
    state_lease = await state.claim("state")
    message = state_lease.messages[0]
    await state.commit(
        state_lease,
        StatePatch(topic=Anchor(message_id=message.id, text="失恋", quote="那次失恋让我难过")),
    )
    _, boundary, lease, context = await request(
        conversation, identity, "以后不要提我的失恋。", "boundary", chat=chat
    )
    assert "失恋" not in context.model_dump_json()
    assert context.companion.repair_action == "applied"
    await conversation.finish(lease)
    for index in range(6):
        _, _, lease, context = await request(
            conversation,
            identity,
            "记得我的跑步吗？",
            f"safe-{index}",
            chat=chat if index < 20 else None,
        )
        assert "失恋" not in context.model_dump_json()
        await conversation.finish(lease)
    # The boundary is a removable user instruction, not a hidden permanent blacklist.
    settings = await Preferences(conversation).read(identity.owner_id, identity.character_id)
    await memory.forget(
        identity.owner_id, settings["source_memory_id"], "forget-boundary", expected_revision=1
    )
    audit = await conversation.companion_explanation(identity.owner_id, boundary.id)
    assert not audit["repairs"]


async def test_actual_mentions_are_recorded_only_when_published_and_replay_is_idempotent(
    conversation,
):
    from hmph.memory import Memory
    from hmph.memory_types import MemoryProposal
    from hmph.model_gateway import SourceMention

    identity = await conversation.bootstrap()
    chat, _, lease, _ = await request(conversation, identity, "我喜欢乌龙茶。", "tea")
    await conversation.finish(lease)
    memory = Memory(conversation.db, digest_key=conversation.digest_key)
    await memory.learn(
        await memory.claim("learn"),
        [MemoryProposal(predicate="preference", value="乌龙茶", quote="我喜欢乌龙茶")],
    )
    source = (await conversation.messages(identity.owner_id, chat.id))[0]
    _, turn, lease, context = await request(
        conversation, identity, "今天还是喜欢泡茶。", "chat", chat=chat
    )
    assert (
        sum(
            d.permission == "mention" and d.reason == "proactive"
            for d in context.companion.memory_use
        )
        <= 1
    )
    hint = (SourceMention(source_id=source.id, quote="乌龙茶"),)
    assert not (await conversation.companion_explanation(identity.owner_id, turn.id))["mentions"]
    await conversation.publish(lease, 0, "又喝了乌", mentions=hint)
    assert not (await conversation.companion_explanation(identity.owner_id, turn.id))["mentions"]
    event = await conversation.publish(lease, 1, "龙茶呀。", mentions=hint)
    assert (await conversation.publish(lease, 1, "龙茶呀。", mentions=hint)).id == event.id
    await conversation.cancel(identity.owner_id, turn.id, "cancel")
    assert (
        len((await conversation.companion_explanation(identity.owner_id, turn.id))["mentions"]) == 1
    )
    _, _, lease, context = await request(
        conversation, identity, "今天还是喜欢泡茶。", "again", chat=chat
    )
    audit = await conversation.companion_explanation(identity.owner_id, lease.turn_id)
    assert (
        next(d for d in audit["plan"]["memory_use"] if d["source_id"] == str(source.id))["reason"]
        == "repeated"
    )
    await conversation.finish(lease)


async def test_forgetting_one_natural_preference_preserves_independently_sourced_settings(
    conversation,
):
    from hmph.memory import Memory
    from hmph.preference_types import ExpressionPreferences

    identity = await conversation.bootstrap()
    chat = await conversation.create_chat(identity.owner_id, identity.character_id, "chat")
    await Preferences(conversation).update(
        identity.owner_id,
        chat.id,
        "settings",
        1,
        ExpressionPreferences(nickname="小木", playfulness="low"),
    )
    _, _, lease, context = await request(
        conversation, identity, "以后先听我说。", "listen", chat=chat
    )
    assert context.preferences.nickname == "小木" and context.preferences.support_mode == "listen"
    await conversation.finish(lease)
    settings = await Preferences(conversation).read(identity.owner_id, identity.character_id)
    await Memory(conversation.db, digest_key=conversation.digest_key).forget(
        identity.owner_id, settings["source_memory_id"], "forget", expected_revision=1
    )
    current = await Preferences(conversation).read(identity.owner_id, identity.character_id)
    assert current["preferences"]["nickname"] == "小木"
    assert current["preferences"]["playfulness"] == "low"
    assert current["preferences"]["support_mode"] == "ask_first"


async def test_ambiguous_repair_clarifies_and_quoted_imperative_does_not_change_settings(
    conversation,
):
    identity = await conversation.bootstrap()
    _, _, lease, context = await request(conversation, identity, "他说“以后叫我小禾”。", "quoted")
    assert context.preferences.nickname is None
    await conversation.finish(lease)
    _, turn, lease, context = await request(conversation, identity, "你又叫错了。", "unclear")
    assert lease.policy_revision == 1
    assert context.companion.act == "clarify" and context.companion.repair_action == "clarify"
    assert (await conversation.companion_explanation(identity.owner_id, turn.id))["repairs"][0][
        "status"
    ] == "open"
    await conversation.finish(lease)
