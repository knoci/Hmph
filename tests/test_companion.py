"""Companion behavior through the existing turn, settings and forget interfaces."""

import pytest

from hmph.preferences import Preferences


@pytest.mark.parametrize(
    "error,code",
    [
        ("provider_timeout", "provider_timeout"),
        ("PRIVATE untrusted provider body", "provider_protocol"),
    ],
)
async def test_annotation_unavailable_records_code_without_discarding_reply(
    conversation, error, code
):
    import json

    from hmph.deepseek import ProviderError
    from hmph.model_gateway import ModelDelta
    from hmph.worker import Worker

    identity = await conversation.bootstrap()
    chat = await conversation.create_chat(identity.owner_id, identity.character_id, "guard")
    turn = await conversation.accept(
        identity.owner_id, chat.id, "query", "你好。", model_mode="deepseek"
    )

    class Model:
        async def stream(self, request):
            yield ModelDelta(text="你好。")

    async def review(request, candidate):
        raise ProviderError(error)

    await Worker(conversation, Model(), model_mode="deepseek", reviewer=review).run_once()
    result = await conversation.get_turn(identity.owner_id, turn.id)
    assert result.status == "completed" and result.error_code is None
    assert result.usage["guard_error_code"] == code
    assert "PRIVATE" not in json.dumps(result.usage)
    assert (await conversation.messages(identity.owner_id, chat.id))[-1].content == "你好。"


@pytest.mark.parametrize(
    "query,recall",
    [
        ("我现在午后固定喝什么？只回答饮品。", True),
        ("我目前每天早上吃什么？", True),
        ("我现在应该喝什么？", False),
        ("我想知道你现在喝什么？", False),
        ("我现在用什么键盘？", True),
        ("我目前在用哪款电脑？", True),
        ("我现在养的是哪只猫？", True),
        ("我应该买什么键盘？", False),
        ("如果我现在用机械键盘，哪款更合适？", False),
        ("What is the name of Devon's dog?", True),
        ("What should Devon's dog eat?", False),
        ("If Devon's dog needed a bed, which would fit?", False),
        ("我参加合唱排练是哪一天？", True),
    ],
)
def test_direct_present_personal_question_is_recall_but_purchase_and_hypothesis_are_not(
    query, recall
):
    from uuid import uuid4

    from hmph.companion_policy import SourceCandidate, decide

    source = SourceCandidate(uuid4(), "我在用青石牌键盘。")
    plan = decide(query, {}, [source], policy_revision=1)
    assert (plan.act == "recall") == recall


@pytest.mark.parametrize(
    "damage,reason",
    [
        ("quote", "source_quote"),
        ("dependency", "source_dependency"),
        ("revision", "stale_lease"),
        ("deleted", "stale_lease"),
    ],
)
async def test_invalid_mention_has_a_specific_reason_and_still_blocks(conversation, damage, reason):
    from sqlalchemy import delete, update

    from hmph.contracts import DomainError
    from hmph.model_gateway import SourceMention
    from hmph.models import Message, TurnSourceUse

    identity = await conversation.bootstrap()
    chat, _, lease, _ = await request(conversation, identity, "我喜欢茉莉茶。", "tea")
    await conversation.finish(lease)
    source = (await conversation.messages(identity.owner_id, chat.id))[0]
    _, _, lease, _ = await request(
        conversation, identity, "你记得我喜欢喝什么？", "recall", chat=chat
    )
    body = "你喜欢柠檬茶。" if damage == "quote" else "你喜欢茉莉茶。"
    hint = SourceMention(source_id=source.id, quote="柠檬茶" if damage == "quote" else "茉莉茶")
    async with conversation.db.sessions.begin() as session:
        if damage == "dependency":
            await session.execute(
                delete(TurnSourceUse).where(
                    TurnSourceUse.turn_id == lease.turn_id, TurnSourceUse.message_id == source.id
                )
            )
        elif damage == "revision":
            await session.execute(update(Message).where(Message.id == source.id).values(revision=2))
        elif damage == "deleted":
            await session.execute(
                update(Message).where(Message.id == source.id).values(deleted=True)
            )
    if reason == "stale_lease":
        with pytest.raises(DomainError, match="stale_lease"):
            await conversation.companion_output_reason(lease, body, (hint,))
        return
    assert await conversation.companion_output_reason(lease, body, (hint,)) == reason
    assert not await conversation.check_companion_output(lease, body, (hint,))


async def test_direct_personal_recall_can_mention_two_valid_sources_after_review(conversation):
    from hmph.memory import Memory
    from hmph.memory_types import MemoryProposal
    from hmph.model_gateway import GuardReview, ModelDelta, SourceMention
    from hmph.worker import Worker

    identity = await conversation.bootstrap()
    chat = await conversation.create_chat(identity.owner_id, identity.character_id, "keyboard")
    memory = Memory(conversation.db, digest_key=conversation.digest_key)
    sources = []
    for i, value in enumerate(("灯塔牌键盘", "青石牌键盘")):
        text = ("我之前用过" if i == 0 else "我现在换成") + value + "。"
        turn = await conversation.accept(identity.owner_id, chat.id, f"source-{i}", text)
        await conversation.cancel(identity.owner_id, turn.id, f"cancel-{i}")
        sources.append((await conversation.messages(identity.owner_id, chat.id))[-1])
        await memory.learn(
            await memory.claim("learn"),
            [MemoryProposal(predicate="preference", value=value, quote=text)],
        )
    turn = await conversation.accept(
        identity.owner_id, chat.id, "recall", "我现在用什么键盘？", model_mode="deepseek"
    )
    body = "你现在用青石牌键盘，之前用的是灯塔牌键盘。"

    class Model:
        async def stream(self, request):
            yield ModelDelta(text=body)

    async def review(request, candidate):
        return GuardReview(
            allowed=True,
            reason="ok",
            mentions=tuple(
                SourceMention(source_id=s.id, quote=value)
                for s, value in zip(sources, ("灯塔牌键盘", "青石牌键盘"), strict=True)
            ),
        )

    await Worker(conversation, Model(), model_mode="deepseek", reviewer=review).run_once()
    actual = (await conversation.messages(identity.owner_id, chat.id))[-1].content
    result = await conversation.get_turn(identity.owner_id, turn.id)
    assert actual == body, result.usage.get("policy_reason")
    assert result.usage["policy_reason"] == "ok"
    assert (
        len((await conversation.companion_explanation(identity.owner_id, turn.id))["mentions"]) == 2
    )


async def test_listening_preference_does_not_veto_generated_speech(conversation):
    identity = await conversation.bootstrap()
    _, _, lease, _ = await request(conversation, identity, "先听我说，不要给建议。", "listen")
    assert (
        await conversation.companion_output_reason(lease, "你应该马上离职。") == "ok"
    )
    assert await conversation.check_companion_output(lease, "你应该马上离职。")


async def test_memory_entry_id_is_diagnosed_and_never_repaired_into_source_id(conversation):
    from hmph.memory import Memory
    from hmph.memory_types import MemoryProposal
    from hmph.model_gateway import SourceMention

    identity = await conversation.bootstrap()
    chat, _, lease, _ = await request(conversation, identity, "我喜欢雪梨汁。", "fact")
    await conversation.finish(lease)
    memory = Memory(conversation.db, digest_key=conversation.digest_key)
    await memory.learn(
        await memory.claim("extract"),
        [MemoryProposal(predicate="preference", value="雪梨汁", quote="我喜欢雪梨汁")],
    )
    fact = (await memory.entries(identity.owner_id, identity.character_id))[0]
    _, _, lease, _ = await request(
        conversation, identity, "你记得我喜欢什么饮品？", "query", chat=chat
    )
    hint = SourceMention(source_id=fact.id, quote="雪梨汁")
    assert (
        await conversation.companion_output_reason(lease, "你喜欢雪梨汁。", (hint,))
        == "memory_id_as_source"
    )
    assert not await conversation.check_companion_output(lease, "你喜欢雪梨汁。", (hint,))


async def test_verified_same_source_witness_survives_a_redundant_translated_hint(conversation):
    from hmph.memory import Memory
    from hmph.memory_types import MemoryProposal
    from hmph.model_gateway import SourceMention

    identity = await conversation.bootstrap()
    chat, _, lease, _ = await request(conversation, identity, "I work at River Glass.", "source")
    await conversation.finish(lease)
    memory = Memory(conversation.db, digest_key=conversation.digest_key)
    await memory.learn(
        await memory.claim("learn"),
        [MemoryProposal(predicate="employer", value="River Glass", quote="I work at River Glass.")],
    )
    source = (await conversation.messages(identity.owner_id, chat.id))[0]
    _, _, lease, _ = await request(
        conversation, identity, "Where do I currently work?", "question", chat=chat
    )
    body = "你在 River Glass 工作。"
    translated = SourceMention(source_id=source.id, quote="River Glass 工作")
    witness = SourceMention(source_id=source.id, quote="River Glass")
    assert await conversation.companion_output_reason(lease, body, (translated,)) == "source_quote"
    assert await conversation.check_companion_output(lease, body, (translated, witness))
    await conversation.publish(lease, 0, body, mentions=(translated, witness))
    assert (
        len(
            (await conversation.companion_explanation(identity.owner_id, lease.turn_id))["mentions"]
        )
        == 1
    )


@pytest.mark.parametrize(
    "original,answer,source_quote,quote",
    [
        ("I volunteer at Cedar Hall.", "你在雪松会馆做志愿者。", "Cedar Hall", "雪松会馆"),
        ("我的演出在昨天。", "演出是在2026年9月12日。", "演出在昨天", "2026年9月12日"),
    ],
)
async def test_reviewed_paraphrase_uses_separate_source_and_published_spans(
    conversation, original, answer, source_quote, quote
):
    from hmph.model_gateway import SourceMention

    identity = await conversation.bootstrap()
    chat, _, lease, _ = await request(conversation, identity, original, "source")
    await conversation.finish(lease)
    source = (await conversation.messages(identity.owner_id, chat.id))[0]
    _, _, lease, _ = await request(
        conversation, identity, "你记得我之前说的事吗？", "recall", chat=chat
    )
    hint = SourceMention(source_id=source.id, quote=quote, source_quote=source_quote)
    assert await conversation.check_companion_output(lease, answer, (hint,))
    # An unpublished span cannot create a mention. The source anchor is checked
    # separately, and a fabricated source anchor still fails closed.
    await conversation.publish(lease, 0, "嗯，", mentions=(hint,))
    assert not (await conversation.companion_explanation(identity.owner_id, lease.turn_id))[
        "mentions"
    ]
    await conversation.publish(lease, 1, answer, mentions=(hint,))
    assert (
        len(
            (await conversation.companion_explanation(identity.owner_id, lease.turn_id))["mentions"]
        )
        == 1
    )
    bad = hint.model_copy(update={"source_quote": "不存在的原话"})
    assert await conversation.companion_output_reason(lease, answer, (bad,)) == "source_quote"


def test_personal_history_enumeration_allows_requested_sources_instead_of_proactive_cap():
    from uuid import uuid4

    from hmph.companion_policy import SourceCandidate, decide

    sources = [
        SourceCandidate(uuid4(), text) for text in ("我去过杭州。", "我去过苏州。", "我去过南京。")
    ]
    plan = decide("我一共去过哪些地方？", {}, sources, policy_revision=1)
    assert all(d.permission == "mention" and d.reason == "requested" for d in plan.memory_use)


def test_event_question_with_matching_content_words_requests_memories_without_first_person():
    from uuid import uuid4

    from hmph.companion_policy import SourceCandidate, decide

    sources = [SourceCandidate(uuid4(), "The sewing-machine technician will visit tomorrow.")]
    question = "On what calendar date was the sewing-machine technician visit scheduled?"
    plan = decide(question, {}, sources, policy_revision=1)
    assert plan.act == "recall" and plan.memory_use[0].reason == "requested"
    advice = decide(
        "Should the sewing-machine technician visit tomorrow?", {}, sources, policy_revision=1
    )
    assert advice.act != "recall"


async def test_longer_replacement_nickname_is_supplied_without_a_speech_veto(
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
    assert await conversation.check_companion_output(lease, "阿禾，收到。")
    await conversation.finish(lease)


async def test_topic_preference_remains_dialogue_context(conversation):
    identity = await conversation.bootstrap()
    chat, _, lease, _ = await request(conversation, identity, "以后叫我小木。", "name")
    await conversation.finish(lease)
    _, _, lease, context = await request(
        conversation, identity, "以后不要提小木。", "topic", chat=chat
    )
    assert context.messages[-1].content == "以后不要提小木。"
    assert context.preferences.nickname == "小木"
    await conversation.finish(lease)


async def request(conversation, identity, text, key, *, chat=None):
    chat = chat or await conversation.create_chat(identity.owner_id, identity.character_id, key)
    turn = await conversation.accept(identity.owner_id, chat.id, key, text)
    lease = await conversation.claim("companion-test")
    return chat, turn, lease, await conversation.generation_request(lease)


async def test_current_controls_and_mention_timing_do_not_erase_assistant_history(conversation):
    identity = await conversation.bootstrap()
    chat, _, lease, _ = await request(conversation, identity, "以后叫我小沐。", "name")
    await conversation.publish(lease, 0, "小沐？行，就这么叫了。")
    await conversation.finish(lease)
    _, _, lease, context = await request(
        conversation, identity, "讲个笑话。", "joke", chat=chat
    )
    assert any(m.role == "assistant" and "就这么叫了" in m.content for m in context.messages)
    await conversation.finish(lease)
    _, _, lease, _ = await request(conversation, identity, "以后叫我小岚。", "rename", chat=chat)
    await conversation.publish(lease, 0, "小岚，那我换个叫法。")
    await conversation.finish(lease)
    _, _, lease, context = await request(
        conversation, identity, "接着聊。", "continue", chat=chat
    )
    assert any(m.role == "assistant" and "换个叫法" in m.content for m in context.messages)
    assert context.preferences.nickname == "小岚"
    assert any(m.role == "assistant" and "就这么叫了" in m.content for m in context.messages)
    await conversation.finish(lease)


@pytest.mark.parametrize(
    "text",
    ["以后请叫我小沐。", "以后请回复简短一点。", "以后请先听我说。"],
)
async def test_current_expression_request_is_not_replaced_by_boundary_notice(conversation, text):
    identity = await conversation.bootstrap()
    chat, _, lease, context = await request(conversation, identity, text, "expression")
    assert context.messages[-1].content == text
    assert (await conversation.messages(identity.owner_id, chat.id))[0].content == text
    await conversation.finish(lease)


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


async def test_short_term_listening_signal_preserves_original_history(conversation):
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
    assert any("失恋" in m.content for m in context.messages)
    assert (
        source.id
        in (await conversation.memory_explanation(identity.owner_id, turn.id))["source_ids"]
    )
    await conversation.finish(lease)


async def test_current_nickname_correction_changes_context_without_replacing_model_reply(
    conversation,
):
    import json

    import httpx
    from pydantic import SecretStr

    from hmph.deepseek import DeepSeekModel
    from hmph.memory import Memory
    from hmph.memory_types import MemoryProposal
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
        assert "小禾" in body["messages"][0]["content"]
        assert "小沐" not in body["messages"][0]["content"]
        assert any(m["content"] == "我叫小沐。" for m in body["messages"][1:])
        assert body["stream"]
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
    await Worker(conversation, model, model_mode="deepseek").run_once()
    assert len(seen) == 1
    assert (await conversation.messages(identity.owner_id, chat.id))[-1].content == "小沐，知道啦。"
    saved = await conversation.get_turn(identity.owner_id, turn.id)
    assert saved.status == "completed" and saved.usage["guard_action"] == "not_requested"


async def test_graph_source_retains_original_excerpt_with_topic_preference(conversation):
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
    assert any("失恋" in m.content for m in context.messages)
    assert any("我喜欢跑步" in s.content for s in context.raw_evidence)
    await service.finish(lease)
    _, _, lease, context = await request(service, identity, "你记得我那次失恋吗？", "allowed")
    assert any("失恋" in s.content for s in context.raw_evidence)
    await service.finish(lease)


async def test_topic_preference_and_source_text_remain_available_to_character(conversation):
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
    assert context.messages[-1].content == "以后不要提我的失恋。"
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
        assert "失恋" in context.model_dump_json()
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
    assert "source_permissions" not in context.companion.model_context()
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
    assert audit["plan"]["memory_use"] == []
    assert any("乌龙茶" in message.content for message in context.messages)
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
