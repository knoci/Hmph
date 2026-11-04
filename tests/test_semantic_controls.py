from uuid import uuid4

import pytest
from sqlalchemy import select

from hmph.dialogue import Dialogue
from hmph.dialogue_types import Anchor, ControlCue, StatePatch
from hmph.memory import Memory
from hmph.models import Message, Pair, Turn
from hmph.preference_types import ExpressionPreferences
from hmph.preferences import Preferences
from hmph.shared_threads import SharedThreads


async def turn(conversation, identity, chat, content):
    accepted = await conversation.accept(identity.owner_id, chat.id, uuid4().hex, content)
    lease = await conversation.claim(uuid4().hex, 30)
    assert lease is not None
    request = await conversation.generation_request(lease)
    await conversation.publish(lease, 0, "嗯，听着呢。")
    await conversation.finish(lease)
    return accepted, request


async def propose(conversation, accepted, content, **cue):
    state = Dialogue(conversation.db)
    lease = await state.claim(uuid4().hex, 30)
    assert lease is not None
    source = next(m for m in lease.messages if m.role == "user" and m.content == content)
    patch = StatePatch(
        controls=[
            ControlCue(
                **cue, evidence=Anchor(message_id=source.id, text=content[:120], quote=content)
            )
        ]
    )
    assert (await state.commit(lease, patch)).status == "committed"
    return source


async def test_semantic_preference_is_fenced_sourced_and_persists_at_1_5_20_and_new_chat(
    conversation,
):
    identity = await conversation.bootstrap()
    chat = await conversation.create_chat(identity.owner_id, identity.character_id, uuid4().hex)
    text = "往后称呼我的时候，用小岚这个名字就好。"
    accepted, _ = await turn(conversation, identity, chat, text)
    source = await propose(
        conversation, accepted, text, kind="preference", field="nickname", value="小岚"
    )
    for i in range(1, 21):
        _, request = await turn(conversation, identity, chat, f"今天第{i}次来聊聊。")
        if i in (1, 5, 20):
            assert request.preferences.nickname == "小岚"
    other = await conversation.create_chat(identity.owner_id, identity.character_id, uuid4().hex)
    _, request = await turn(conversation, identity, other, "我回来了。")
    assert request.preferences.nickname == "小岚"
    async with conversation.db.sessions() as session:
        origin = await session.get(Turn, accepted.id)
        assert origin.companion_plan["semantic_control"]["status"] == "applied"
    await Memory(conversation.db, digest_key=conversation.digest_key).forget_source(
        identity.owner_id, source.id, uuid4().hex, expected_revision=source.revision
    )
    _, request = await turn(conversation, identity, other, "接着聊。")
    assert request.preferences.nickname is None


async def test_background_control_never_invalidates_inflight_or_overwrites_newer_settings(
    conversation,
):
    identity = await conversation.bootstrap()
    chat = await conversation.create_chat(identity.owner_id, identity.character_id, uuid4().hex)
    text = "往后你说话可以短一些，信息太多我看着累。"
    accepted = await conversation.accept(identity.owner_id, chat.id, uuid4().hex, text)
    lease = await conversation.claim(uuid4().hex, 30)
    source = await propose(
        conversation, accepted, text, kind="preference", field="reply_length", value="short"
    )
    await conversation.generation_request(lease)
    await conversation.publish(lease, 0, "好，简短些。")
    await conversation.finish(lease)
    service = Preferences(conversation)
    current = await service.read(identity.owner_id, identity.character_id)
    await service.update(
        identity.owner_id,
        chat.id,
        uuid4().hex,
        current["revision"],
        ExpressionPreferences(reply_length="long"),
    )
    _, request = await turn(conversation, identity, chat, "接着聊。")
    assert request.preferences.reply_length == "long"
    async with conversation.db.sessions() as session:
        origin = await session.get(Turn, accepted.id)
        assert origin.companion_plan["semantic_control"]["status"] == "stale"
        assert not (await session.get(Message, source.id)).deleted


async def test_implicit_shared_invitation_progress_pause_resume_and_ambiguous_target(conversation):
    identity = await conversation.bootstrap()
    chat = await conversation.create_chat(identity.owner_id, identity.character_id, uuid4().hex)
    shared = SharedThreads(conversation)
    steps = [
        ("以后陪我一起跟进法语考试，好不好？", "create", "法语考试", "open"),
        ("法语考试的第一套模拟题我已经做完了。", "progress", "法语考试", "open"),
        ("法语考试先搁一搁吧，这阵子别继续追问了。", "pause", "法语考试", "paused"),
        ("我们继续法语考试这件事吧。", "resume", "法语考试", "open"),
    ]
    sources = []
    for text, operation, target, status in steps:
        accepted, _ = await turn(conversation, identity, chat, text)
        sources.append(
            await propose(
                conversation, accepted, text, kind="thread", operation=operation, target=target
            )
        )
        await turn(conversation, identity, chat, "我回来了。")
        items = (await shared.read(identity.owner_id, identity.character_id))["items"]
        assert len(items) == 1 and items[0]["status"] == status
    rows = await shared.sources(identity.owner_id, items[0]["id"])
    assert len(rows["items"]) == 4
    await shared.write(identity.owner_id, chat.id, uuid4().hex, topic="法语考试口语准备")
    text = "先暂停法语考试吧。"
    accepted, _ = await turn(conversation, identity, chat, text)
    await propose(conversation, accepted, text, kind="thread", operation="pause", target="法语考试")
    _, request = await turn(conversation, identity, chat, "我回来了。")
    assert request.companion.act == "clarify"
    assert all(
        t["status"] == "open"
        for t in (await shared.read(identity.owner_id, identity.character_id))["items"]
    )
    await Memory(conversation.db, digest_key=conversation.digest_key).forget_source(
        identity.owner_id, sources[0].id, uuid4().hex, expected_revision=sources[0].revision
    )
    remaining = (await shared.read(identity.owner_id, identity.character_id))["items"]
    assert all(t["id"] != items[0]["id"] for t in remaining)


@pytest.mark.parametrize(
    "text,value",
    [
        ("朋友说以后叫我小岚。", "小岚"),
        ("如果以后叫我小岚，会不会怪？", "小岚"),
        ("以后称呼我小岚就好。", "凭空编的称呼"),
    ],
)
async def test_unacceptable_semantic_control_is_not_staged(conversation, text, value):
    identity = await conversation.bootstrap()
    chat = await conversation.create_chat(identity.owner_id, identity.character_id, uuid4().hex)
    accepted, _ = await turn(conversation, identity, chat, text)
    await propose(conversation, accepted, text, kind="preference", field="nickname", value=value)
    async with conversation.db.sessions() as session:
        origin = await session.get(Turn, accepted.id)
        assert "semantic_control" not in (origin.companion_plan or {})
        pair = await session.scalar(select(Pair).where(Pair.owner_id == identity.owner_id))
        assert pair.preferences.get("nickname") is None


async def test_pending_boundary_forget_never_applies_and_new_boundary_filters_all_channels(
    conversation,
):
    identity = await conversation.bootstrap()
    chat = await conversation.create_chat(identity.owner_id, identity.character_id, uuid4().hex)
    text = "往后请不要主动聊起松湾项目，我想先缓一缓。"
    accepted, _ = await turn(conversation, identity, chat, text)
    source = await propose(
        conversation, accepted, text, kind="boundary", target="松湾项目", mode="on_request"
    )
    memory = Memory(conversation.db, digest_key=conversation.digest_key)
    await memory.forget_source(identity.owner_id, source.id, uuid4().hex, expected_revision=1)
    _, request = await turn(conversation, identity, chat, "我回来了。")
    async with conversation.db.sessions() as session:
        pair = await session.scalar(select(Pair).where(Pair.owner_id == identity.owner_id))
        assert not pair.preferences.get("_boundaries")
    accepted, _ = await turn(conversation, identity, chat, text)
    await propose(
        conversation, accepted, text, kind="boundary", target="松湾项目", mode="on_request"
    )
    _, request = await turn(conversation, identity, chat, "我回来了。")
    assert "松湾项目" not in request.model_dump_json()


async def test_natural_pause_suppresses_question_before_background_commit(conversation):
    identity = await conversation.bootstrap()
    chat = await conversation.create_chat(identity.owner_id, identity.character_id, uuid4().hex)
    await SharedThreads(conversation).write(
        identity.owner_id, chat.id, uuid4().hex, topic="法语考试"
    )
    _, request = await turn(
        conversation, identity, chat, "法语考试先搁一搁吧，这阵子别继续追问了。"
    )
    assert request.companion.shared_thread is None
    assert request.companion.shared_thread_action == "pause"
    assert request.companion.rhythm["question_limit"] == 0


async def test_semantic_progress_is_retrievable_and_failed_bundle_rolls_back(conversation):
    identity = await conversation.bootstrap()
    chat = await conversation.create_chat(identity.owner_id, identity.character_id, uuid4().hex)
    shared = SharedThreads(conversation)
    item = await shared.write(identity.owner_id, chat.id, uuid4().hex, topic="法语考试")
    text = "法语考试的第一套模拟题我已经做完了。"
    accepted, _ = await turn(conversation, identity, chat, text)
    source = await propose(
        conversation, accepted, text, kind="thread", operation="progress", target="法语考试"
    )
    _, request = await turn(conversation, identity, chat, "法语考试我做到哪里了？")
    assert any(r.message_id == source.id for r in request.raw_evidence)
    assert any(
        d.source_id == source.id and d.permission == "mention" for d in request.companion.memory_use
    )
    current = (await shared.read(identity.owner_id, identity.character_id))["items"][0]
    await shared.write(
        identity.owner_id,
        chat.id,
        uuid4().hex,
        thread_id=item["id"],
        expected_revision=current["revision"],
        status="paused",
    )
    text = "往后叫我小岚；法语考试的第二套题也完成了。"
    accepted, _ = await turn(conversation, identity, chat, text)
    dialogue = Dialogue(conversation.db)
    lease = await dialogue.claim(uuid4().hex, 30)
    source = next(m for m in lease.messages if m.content == text)
    evidence = Anchor(message_id=source.id, text=text, quote=text)
    await dialogue.commit(
        lease,
        StatePatch(
            controls=[
                ControlCue(kind="preference", field="nickname", value="小岚", evidence=evidence),
                ControlCue(
                    kind="thread", operation="progress", target="法语考试", evidence=evidence
                ),
            ]
        ),
    )
    _, request = await turn(conversation, identity, chat, "我回来了。")
    assert request.preferences.nickname is None
    assert request.companion.act == "clarify"
    assert (await shared.read(identity.owner_id, identity.character_id))["items"][0][
        "status"
    ] == "paused"


async def test_updater_receives_thread_status_and_changed_policy_cannot_queue_controls(
    conversation,
):
    identity = await conversation.bootstrap()
    chat = await conversation.create_chat(identity.owner_id, identity.character_id, uuid4().hex)
    shared = SharedThreads(conversation)
    item = await shared.write(identity.owner_id, chat.id, uuid4().hex, topic="法语考试")
    await shared.write(
        identity.owner_id,
        chat.id,
        uuid4().hex,
        thread_id=item["id"],
        expected_revision=item["revision"],
        status="paused",
    )
    text = "我们继续法语考试这件事吧。"
    accepted, _ = await turn(conversation, identity, chat, text)
    dialogue = Dialogue(conversation.db)
    lease = await dialogue.claim(uuid4().hex, 30)
    assert [(t.topic, t.status) for t in lease.shared_threads] == [("法语考试", "paused")]
    source = next(m for m in lease.messages if m.content == text)
    prefs = Preferences(conversation)
    current = await prefs.read(identity.owner_id, identity.character_id)
    await prefs.update(
        identity.owner_id,
        chat.id,
        uuid4().hex,
        current["revision"],
        ExpressionPreferences(nickname="小夏"),
    )
    result = await dialogue.commit(
        lease,
        StatePatch(
            controls=[
                ControlCue(
                    kind="thread",
                    operation="resume",
                    target="法语考试",
                    evidence=Anchor(message_id=source.id, text=text, quote=text),
                )
            ]
        ),
    )
    assert result.status == "committed"
    async with conversation.db.sessions() as session:
        origin = await session.get(Turn, accepted.id)
        assert "semantic_control" not in origin.companion_plan
