"""Product evidence boundary: native IDs are untrusted, PostgreSQL owns text."""

from types import SimpleNamespace
from uuid import uuid4

import pytest

from hmph.contracts import DomainError
from hmph.memory_builds import MemoryBuilds
from hmph.models import MemoryNativeSource
from hmph.online_memory import MemorySelection


async def say(conversation, identity, text, key):
    chat = await conversation.create_chat(identity.owner_id, identity.character_id, key)
    turn = await conversation.accept(identity.owner_id, chat.id, key, text)
    await conversation.cancel(identity.owner_id, turn.id, key)
    return (await conversation.messages(identity.owner_id, chat.id))[0]


async def ready_graph(conversation, identity):
    builds = MemoryBuilds(conversation.db, digest_key=conversation.digest_key)
    build = await builds.request(
        identity.owner_id, identity.character_id, uuid4().hex, backend="graphiti", force=True
    )
    lease = await builds.claim("test-writer", backend="graphiti")
    ids = [str(uuid4()) for _ in lease.sources]
    await builds.register_native_sources(
        lease,
        [
            MemoryNativeSource(
                space_id=lease.space_id,
                native_id=native,
                message_id=s.message_id,
                message_revision=s.revision,
                source_order=s.source_order,
            )
            for s, native in zip(lease.sources, ids, strict=True)
        ],
    )
    await builds.quiesce(lease)
    assert await builds.publish(lease)
    return build, lease, ids


async def test_explicit_selection_is_versioned_and_does_not_follow_latest_ready(conversation):
    identity = await conversation.bootstrap()
    await say(conversation, identity, "我现在住在杭州。", "home")
    selection = MemorySelection(conversation.db, digest_key=conversation.digest_key)
    initial = await selection.get(identity.owner_id, identity.character_id)
    assert initial.mode == "lexical" and initial.revision == 1
    first, _, _ = await ready_graph(conversation, identity)
    assert (await selection.get(identity.owner_id, identity.character_id)) == initial
    chosen = await selection.update(
        identity.owner_id, identity.character_id, "select", expected_revision=1, build_id=first.id
    )
    assert chosen.mode == "graphiti_experimental" and chosen.revision == 2
    assert chosen.build_id == first.id
    with pytest.raises(DomainError, match="memory_config_conflict"):
        await selection.update(
            identity.owner_id, identity.character_id, "stale", expected_revision=1, build_id=None
        )
    await ready_graph(conversation, identity)
    unchanged = await selection.get(identity.owner_id, identity.character_id)
    assert unchanged.build_id == first.id and unchanged.revision == 2
    assert not unchanged.available
    # Replaying an old settings write cannot restore a retired build.
    assert (
        await selection.update(
            identity.owner_id,
            identity.character_id,
            "select",
            expected_revision=1,
            build_id=first.id,
        )
        == unchanged
    )
    baseline = await selection.update(
        identity.owner_id, identity.character_id, "baseline", expected_revision=2, build_id=None
    )
    assert baseline.mode == "lexical" and baseline.revision == 3 and baseline.available


class NativeReader:
    """Substitutes only the external framework/process boundary."""

    def __init__(self, ids):
        self.ids = ids

    async def retrieve_sources(self, ticket, query):
        from hmph.online_memory import NativeSources

        return NativeSources(source_ids=tuple(self.ids))


@pytest.mark.parametrize("same_chat", [True, False])
def test_current_state_admits_the_reply_between_on_topic_sources(same_chat):
    from hmph.online_memory import _prefer_bridged_update

    chat = uuid4()
    question = SimpleNamespace(
        conversation_id=chat, source_order=10, content="Do I still need the yellow cycling pass?"
    )
    answer = SimpleNamespace(
        conversation_id=chat if same_chat else uuid4(),
        source_order=11, content="No, they stopped checking last week."
    )
    confirmation = SimpleNamespace(
        conversation_id=chat, source_order=12,
        content="Right, the yellow cycling pass is no longer required."
    )
    ranked = [question, confirmation, answer]
    expected = [answer, question, confirmation] if same_chat else ranked
    assert _prefer_bridged_update(ranked, question.content) == expected


@pytest.mark.parametrize("historical", [False, True])
async def test_workspace_links_withdrawal_only_for_current_question(conversation, historical):
    identity = await conversation.bootstrap()
    owner, character = identity.owner_id, identity.character_id
    prior = await say(conversation, identity, "可以把灯塔照片放进群相册。", "permission")
    builds = MemoryBuilds(conversation.db, digest_key=conversation.digest_key)
    requested = await builds.request(owner, character, "build", backend="mem0_timem")
    writer = await builds.claim("writer", backend="mem0_timem")
    await builds.quiesce(writer)
    assert await builds.publish(writer)
    selection = MemorySelection(
        conversation.db, digest_key=conversation.digest_key, allow_timem=True
    )
    await selection.update(owner, character, "select", expected_revision=1, build_id=requested.id)
    change = await say(conversation, identity, "灯塔照片从群相册里撤掉。", "withdrawal")
    conversation.source_reader = NativeReader([prior.id])
    chat = await conversation.create_chat(owner, character, "recall")
    when = "当时" if historical else "现在"
    await conversation.accept(owner, chat.id, "question", f"{when}灯塔照片能放群相册吗？")
    request = await conversation.generation_request(await conversation.claim("reader"))
    assert {prior.id, change.id} <= {s.message_id for s in request.raw_evidence}
    links = request.workspace.withdrawal_evidence
    if historical:
        assert not links
    else:
        assert [(link.prior_source_id, link.withdrawal_source_id) for link in links] == [
            (prior.id, change.id)
        ]


@pytest.mark.parametrize("condition_before_change", [False, True])
async def test_current_withdrawal_keeps_exact_scope_and_condition(
    conversation, condition_before_change
):
    identity = await conversation.bootstrap()
    owner, character = identity.owner_id, identity.character_id
    other = await say(conversation, identity, "便签还是喜欢紫色。", "other-preference")
    action = (
        "晚间喝水提醒作废。"
        if condition_before_change
        else "晚间喝水提醒撤掉。除非我自己请你提醒。"
    )
    body = f"紫色便签可以留着，但{action}"
    condition_body = (
        "如果我确实需要，到时候会告诉你。晚间喝水提醒先停。"
        if condition_before_change
        else "以后除非我主动说，不用问了。"
    )
    source_chat = await conversation.create_chat(owner, character, "changes")
    changes = [("scoped-change", body), ("condition", condition_body)]
    if condition_before_change:
        changes.reverse()
    for key, content in changes:
        turn = await conversation.accept(owner, source_chat.id, key, content)
        await conversation.cancel(owner, turn.id, key)
    messages = await conversation.messages(owner, source_chat.id)
    change, condition = reversed(messages) if condition_before_change else messages
    builds = MemoryBuilds(conversation.db, digest_key=conversation.digest_key)
    requested = await builds.request(owner, character, "build", backend="mem0_timem")
    writer = await builds.claim("writer", backend="mem0_timem")
    await builds.quiesce(writer)
    assert await builds.publish(writer)
    selection = MemorySelection(
        conversation.db, digest_key=conversation.digest_key, allow_timem=True
    )
    await selection.update(owner, character, "select", expected_revision=1, build_id=requested.id)
    conversation.source_reader = NativeReader([other.id, change.id, condition.id])
    chat = await conversation.create_chat(owner, character, "recall")
    query = (
        "晚间喝水提醒能不能继续用？"
        if condition_before_change
        else "现在还应该主动给我晚间喝水提醒吗？"
    )
    await conversation.accept(owner, chat.id, "q", query)
    request = await conversation.generation_request(await conversation.claim("reader"))
    sources = {source.message_id: source for source in request.raw_evidence}
    assert set(sources) == {change.id, condition.id}
    source, followup = sources[change.id], sources[condition.id]
    assert source.content == action
    assert source.content == body[source.start_offset:source.end_offset]
    assert followup.content == condition_body
    assert not request.workspace.withdrawal_evidence


async def test_timem_refresh_keeps_the_selected_snapshot_until_explicit_switch(conversation):
    from hmph.conversation import Conversation

    identity = await conversation.bootstrap()
    owner, character = identity.owner_id, identity.character_id
    original = await say(conversation, identity, "我在山脚拍到过一只白尾鸟。", "old")
    builds = MemoryBuilds(conversation.db, digest_key=conversation.digest_key)

    async def publish(key):
        build = await builds.request(owner, character, key, backend="mem0_timem", force=True)
        writer = await builds.claim(key, backend="mem0_timem")
        await builds.quiesce(writer)
        assert await builds.publish(writer)
        return build.id

    first = await publish("first")
    selection = MemorySelection(
        conversation.db, digest_key=conversation.digest_key, allow_timem=True
    )
    active = await selection.update(
        owner, character, "choose-old", expected_revision=1, build_id=first
    )
    service = Conversation(
        conversation.db,
        digest_key=conversation.digest_key,
        source_reader=NativeReader([original.id]),
    )
    chat = await service.create_chat(owner, character, "active-chat")
    await service.accept(owner, chat.id, "lookup", "我在山脚拍到了什么？")
    lease = await service.claim("lookup")
    assert (await service.generation_request(lease)).workspace.backend == "mem0_timem"

    intermediate = await publish("intermediate")
    latest = await publish("latest")
    assert (await builds.current(owner, character, backend="mem0_timem")).id == latest
    assert (await builds.get(owner, intermediate)).status == "retired"
    assert (await selection.get(owner, character)) == active and active.available
    await service.publish(lease, 0, "你拍到过白尾鸟。")
    await service.finish(lease)

    replacement = await selection.update(
        owner, character, "choose-new", expected_revision=active.revision, build_id=latest
    )
    assert replacement.build_id == latest and replacement.available
    assert (await builds.get(owner, first)).status == "retired"


async def test_forgetting_retires_both_ready_timem_snapshots_during_handoff(conversation):
    from hmph.memory import Memory

    identity = await conversation.bootstrap()
    owner, char = identity.owner_id, identity.character_id
    source = await say(conversation, identity, "我常在南桥边散步。", "path")
    builds = MemoryBuilds(conversation.db, digest_key=conversation.digest_key)
    selection = MemorySelection(
        conversation.db, digest_key=conversation.digest_key, allow_timem=True
    )
    ready = []
    for i in range(2):
        key = f"snapshot-{i}"
        build = await builds.request(owner, char, key, backend="mem0_timem", force=True)
        writer = await builds.claim(key, backend="mem0_timem")
        await builds.quiesce(writer)
        assert await builds.publish(writer)
        ready.append(build.id)
        if i == 0:
            await selection.update(owner, char, "select", expected_revision=1, build_id=build.id)

    memory = Memory(conversation.db, digest_key=conversation.digest_key)
    receipt = await memory.forget_source(owner, source.id, "forget", expected_revision=1)
    assert receipt.status == "pending_cleanup"
    for build_id in ready:
        assert (await builds.get(owner, build_id)).status == "retired"
        assert (await builds.spaces(owner, build_id))[-1].status == "retired"
    assert not (await selection.get(owner, char)).available
    assert await builds.current(owner, char, backend="mem0_timem") is None


async def test_new_correction_reaches_generation_when_committed_facts_fill_the_budget(
    conversation,
):
    from hmph.conversation import Conversation
    from hmph.memory import Memory
    from hmph.memory_types import MemoryProposal

    identity = await conversation.bootstrap()
    owner, character = identity.owner_id, identity.character_id
    memory = Memory(conversation.db, digest_key=conversation.digest_key)
    original = None
    for i in range(8):
        value = f"森屿陶艺样品{i}，青灰釉手拉陶杯"
        source = await say(conversation, identity, f"我选了{value}。", f"sample-{i}")
        if original is None:
            original = source
        await memory.learn(
            await memory.claim("extract"),
            [MemoryProposal(predicate=f"sample_{i}", value=value, quote=source.content)],
        )
    builds = MemoryBuilds(conversation.db, digest_key=conversation.digest_key)
    build = await builds.request(owner, character, "build", backend="mem0_timem")
    writer = await builds.claim("combine", backend="mem0_timem")
    await builds.quiesce(writer)
    assert await builds.publish(writer)
    await MemorySelection(
        conversation.db, digest_key=conversation.digest_key, allow_timem=True
    ).update(owner, character, "select", expected_revision=1, build_id=build.id)
    new = await say(
        conversation,
        identity,
        "森屿陶艺工坊改期了，周日下午取件。" + "我还整理了随身的一些工具。" * 40,
        "fresh",
    )
    service = Conversation(
        conversation.db,
        digest_key=conversation.digest_key,
        source_reader=NativeReader([original.id]),
    )
    chat = await service.create_chat(owner, character, "follow-up")
    await service.accept(owner, chat.id, "ask", "森屿陶艺的陶杯什么时候去取？")
    lease = await service.claim("lookup")
    request = await service.generation_request(lease)
    assert any("森屿陶艺" in hit.value for hit in request.evidence)
    assert new.id in {source.message_id for source in request.raw_evidence}
    await service.finish(lease)


@pytest.mark.parametrize("backend", ["mem0", "mem0_timem"])
async def test_mem0_keeps_unrelated_current_fact_after_forget_retires_build(conversation, backend):
    from hmph.conversation import Conversation
    from hmph.memory import Memory
    from hmph.memory_build_types import CleanupProof
    from hmph.memory_types import MemoryProposal

    identity = await conversation.bootstrap()
    owner, character = identity.owner_id, identity.character_id
    original = await say(conversation, identity, "我在远山设计工作。", "work")
    memory = Memory(conversation.db, digest_key=conversation.digest_key)
    await memory.learn(
        await memory.claim("extract"),
        [MemoryProposal(predicate="employer", value="远山设计", quote=original.content)],
    )
    poem = await say(conversation, identity, "我正在背山居秋暝。", "poem")
    await say(conversation, identity, "周末小幅版画试印。", "unrelated-thread")
    await say(conversation, identity, "小幅版画的试印现在是什么状态？", "unrelated-question")
    builds = MemoryBuilds(conversation.db, digest_key=conversation.digest_key)
    build = await builds.request(owner, character, "build", backend=backend)
    build_lease = await builds.claim("native", backend=backend)
    await builds.quiesce(build_lease)
    assert await builds.publish(build_lease)
    selection = MemorySelection(
        conversation.db, digest_key=conversation.digest_key, allow_mem0=True, allow_timem=True
    )
    await selection.update(owner, character, "select", expected_revision=1, build_id=build.id)
    service = Conversation(
        conversation.db, digest_key=conversation.digest_key, source_reader=NativeReader([poem.id])
    )
    for key in ("ready", "after-forget"):
        chat = await service.create_chat(owner, character, key)
        await service.accept(owner, chat.id, key, "我现在在哪工作？")
        lease = await service.claim("reader")
        request = await service.generation_request(lease)
        assert any(hit.value == "远山设计" for hit in request.evidence)
        if key == "after-forget":
            assert request.workspace.retrieval_status == "build_unavailable"
            assert "山居秋暝" not in request.model_dump_json()
            if backend == "mem0_timem":
                assert "小幅版画" not in request.model_dump_json()
        await service.finish(lease)
        if key == "ready":
            receipt = await memory.forget_source(owner, poem.id, "forget", expected_revision=1)
            cleanup = await builds.claim_cleanup("cleanup", backend=backend)
            await builds.finish_cleanup(
                cleanup, CleanupProof(store_empty=True, auxiliary_empty=True)
            )
            assert (await memory.deletion(owner, receipt.id)).status == "completed"


async def test_successful_graph_result_is_complemented_by_matching_canonical_original(conversation):
    from hmph.conversation import Conversation

    identity = await conversation.bootstrap()
    important = await say(conversation, identity, "我前天参加了合唱排练。", "event")
    await say(conversation, identity, "我把蓝色雨伞放在玄关。", "unrelated")
    build, _, native = await ready_graph(conversation, identity)
    await MemorySelection(conversation.db, digest_key=conversation.digest_key).update(
        identity.owner_id, identity.character_id, "select", expected_revision=1, build_id=build.id
    )
    service = Conversation(
        conversation.db, digest_key=conversation.digest_key, source_reader=NativeReader(native[1:])
    )
    chat = await service.create_chat(identity.owner_id, identity.character_id, "query")
    await service.accept(identity.owner_id, chat.id, "query", "我参加合唱排练是哪一天？")
    lease = await service.claim("reader")
    req = await service.generation_request(lease)
    assert important.id in {s.message_id for s in req.raw_evidence}
    assert req.workspace.backend == "graphiti"
    assert req.workspace.candidate_count <= 80
    assert len(req.raw_evidence) <= 12
    explanation = await service.memory_explanation(identity.owner_id, lease.turn_id)
    assert explanation["context"]["lexical_complement_sources"] == 1
    await service.finish(lease)


async def test_mapped_foreign_pair_source_is_rejected_even_if_native_id_exists(conversation):
    from sqlalchemy import update

    from hmph.conversation import Conversation
    from hmph.models import MemoryBuildSource

    identity = await conversation.bootstrap()
    own = await say(conversation, identity, "我现在住在杭州。", "own")
    build, writer, native = await ready_graph(conversation, identity)
    await MemorySelection(conversation.db, digest_key=conversation.digest_key).update(
        identity.owner_id, identity.character_id, "select", expected_revision=1, build_id=build.id
    )
    outsider = await conversation.bootstrap(owner_id=uuid4())
    secret = await say(conversation, outsider, "我现在住在海口。", "other")
    # Corrupt external provenance must not bypass the authoritative pair check.
    async with conversation.db.sessions.begin() as session:
        session.add(
            MemoryBuildSource(
                build_id=build.id, message_id=secret.id, message_revision=1, source_order=1
            )
        )
        await session.execute(
            update(MemoryNativeSource)
            .where(MemoryNativeSource.space_id == writer.space_id)
            .values(message_id=secret.id)
        )
    service = Conversation(
        conversation.db, digest_key=conversation.digest_key, source_reader=NativeReader(native)
    )
    chat = await service.create_chat(identity.owner_id, identity.character_id, "query")
    await service.accept(identity.owner_id, chat.id, "query", "我现在住在哪？")
    lease = await service.claim("reader")
    request = await service.generation_request(lease)
    assert request.workspace.retrieval_status == "invalid_sources"
    assert [s.message_id for s in request.raw_evidence] == [own.id]
    assert secret.content not in request.model_dump_json()
    await service.finish(lease)


async def test_online_context_uses_canonical_sources_and_cross_session_correction(conversation):
    from hmph.conversation import Conversation

    identity = await conversation.bootstrap()
    original = await say(conversation, identity, "我现在住在杭州。", "old")
    build, _, native = await ready_graph(conversation, identity)
    selection = MemorySelection(conversation.db, digest_key=conversation.digest_key)
    await selection.update(
        identity.owner_id, identity.character_id, "select", expected_revision=1, build_id=build.id
    )
    correction = await say(conversation, identity, "更正，我已经搬到苏州，现在不住杭州了。", "new")
    service = Conversation(
        conversation.db, digest_key=conversation.digest_key, source_reader=NativeReader(native)
    )
    chat = await service.create_chat(identity.owner_id, identity.character_id, "query")
    accepted = await service.accept(identity.owner_id, chat.id, "query", "我现在住在哪？")
    lease = await service.claim("reader")
    request = await service.generation_request(lease)
    assert not request.evidence  # Never send framework assertions as evidence.
    assert {s.message_id for s in request.raw_evidence} == {original.id, correction.id}
    assert {s.content for s in request.raw_evidence} == {original.content, correction.content}
    assert request.workspace.backend == "graphiti"
    assert request.workspace.retrieval_status == "ok"
    assert request.workspace.delta_gap == 0
    explanation = await service.memory_explanation(identity.owner_id, accepted.id)
    assert {original.id, correction.id} <= set(explanation["source_ids"])
    assert explanation["context"]["build_watermark"] == 1
    assert explanation["context"]["cutoff"] == 3
    await service.publish(lease, 0, "你已经搬到苏州了。")
    await service.finish(lease)


@pytest.mark.parametrize("failure", ["foreign_id", "crash", "timeout", "empty"])
async def test_retrieval_failure_uses_legal_raw_baseline_and_keeps_reason(conversation, failure):
    import asyncio

    from hmph.conversation import Conversation
    from hmph.online_memory import NativeSources

    identity = await conversation.bootstrap()
    original = await say(conversation, identity, "我现在住在杭州。", "home")
    build, _, _ = await ready_graph(conversation, identity)
    await MemorySelection(conversation.db, digest_key=conversation.digest_key).update(
        identity.owner_id, identity.character_id, "select", expected_revision=1, build_id=build.id
    )

    class BrokenReader:
        async def retrieve_sources(self, ticket, query):
            if failure == "crash":
                raise RuntimeError("external error with private input")
            if failure == "timeout":
                await asyncio.sleep(10)
            return NativeSources(source_ids=(uuid4(),) if failure == "foreign_id" else ())

    service = Conversation(
        conversation.db, digest_key=conversation.digest_key, source_reader=BrokenReader()
    )
    chat = await service.create_chat(identity.owner_id, identity.character_id, "query")
    await service.accept(identity.owner_id, chat.id, "query", "我现在住在哪？")
    lease = await service.claim("reader")
    request = await service.generation_request(lease)
    expected = {
        "foreign_id": "invalid_sources",
        "crash": "reader_failed",
        "timeout": "timeout",
        "empty": "empty",
    }[failure]
    assert request.workspace.retrieval_status == expected
    assert request.raw_evidence[0].message_id == original.id
    assert not request.evidence
    await service.finish(lease)


async def test_delta_overflow_cannot_reintroduce_old_graph_value(conversation):
    from hmph.conversation import Conversation

    identity = await conversation.bootstrap()
    old = await say(conversation, identity, "我现在住在杭州。", "old")
    build, _, native = await ready_graph(conversation, identity)
    await MemorySelection(conversation.db, digest_key=conversation.digest_key).update(
        identity.owner_id, identity.character_id, "select", expected_revision=1, build_id=build.id
    )
    await say(
        conversation, identity, "更正，我已经搬到苏州。" + "这是一段很长的补充说明。" * 160, "new"
    )
    service = Conversation(
        conversation.db, digest_key=conversation.digest_key, source_reader=NativeReader(native)
    )
    chat = await service.create_chat(identity.owner_id, identity.character_id, "query")
    await service.accept(identity.owner_id, chat.id, "query", "我现在住在哪？")
    lease = await service.claim("reader")
    request = await service.generation_request(lease)
    assert request.workspace.delta_gap == 1 and request.workspace.coverage == "partial"
    assert old.id not in {s.message_id for s in request.raw_evidence}
    assert not request.evidence
    await service.finish(lease)


@pytest.mark.parametrize(
    ("filler_count", "correction_kind"),
    [(14, "short"), (14, "padded"), (14, "spread"), (100, "short"), (100, "spread")],
)
async def test_timem_keeps_older_evidence_through_unrelated_backlog(
    conversation, filler_count, correction_kind
):
    from hmph.conversation import Conversation

    identity = await conversation.bootstrap()
    owner, character = identity.owner_id, identity.character_id
    old = await say(conversation, identity, "森屿陶艺工坊的小碗是雾蓝色，周六取。", "old")
    builds = MemoryBuilds(conversation.db, digest_key=conversation.digest_key)
    build = await builds.request(owner, character, "build", backend="mem0_timem")
    writer = await builds.claim("index", backend="mem0_timem")
    await builds.quiesce(writer)
    assert await builds.publish(writer)
    await MemorySelection(
        conversation.db, digest_key=conversation.digest_key, allow_timem=True
    ).update(owner, character, "select", expected_revision=1, build_id=build.id)
    update = "工坊改期了，周日下午去拿。"
    correction_body = update
    if correction_kind != "short":
        correction_body += "补充很长的说明。" * 200
    if correction_kind == "spread":
        correction_body += "工坊又通知改成星期一了。"
    correction = await say(conversation, identity, correction_body, "correction")
    for index in range(filler_count):
        await say(
            conversation, identity, f"这是第{index}次海边慢跑，我听了一路摇滚。", f"run-{index}"
        )

    class RecoveringReader(NativeReader):
        recovering = False

        async def retrieve_sources(self, ticket, query):
            from hmph.online_memory import SourceRetrievalError

            if self.recovering:
                raise SourceRetrievalError("reader_recovering")
            return await super().retrieve_sources(ticket, query)

    reader = RecoveringReader([old.id])
    service = Conversation(
        conversation.db, digest_key=conversation.digest_key, source_reader=reader
    )
    chat = await service.create_chat(owner, character, "new-chat")
    turn = await service.accept(owner, chat.id, "lookup", "森屿陶艺的小碗是什么颜色，什么时候取？")
    lease = await service.claim("lookup")
    request = await service.generation_request(lease)
    explanation = await service.memory_explanation(owner, turn.id)
    audit = explanation["context"]
    if filler_count > 80:
        assert audit["delta_scope"] == "matched"
        assert audit["delta_scanned"] == 1
        assert audit["delta_scanned_from_order"] is None
        assert audit["delta_unselected"] is None
    else:
        assert audit["delta_scope"] == "recent"
        assert audit["delta_scanned"] == 15
        assert audit["delta_unselected"] == 14
    assert request.workspace.coverage == "partial"
    source_ids = {s.message_id for s in request.raw_evidence}
    if correction_kind == "spread":
        assert source_ids == set()
        assert request.workspace.delta_gap == 1
    else:
        assert source_ids == {old.id, correction.id}
        assert request.workspace.delta_gap == 0
        if correction_kind == "padded":
            selected = next(s for s in request.raw_evidence if s.message_id == correction.id)
            assert selected.truncated and update in selected.content
            assert selected.content == correction_body[selected.start_offset : selected.end_offset]
    await service.finish(lease)

    if filler_count > 80 and correction_kind == "short":
        reader.recovering = True
        await service.accept(owner, chat.id, "during-recovery", "森屿陶艺取件要约哪天？")
        lease = await service.claim("recovering")
        request = await service.generation_request(lease)
        assert request.workspace.retrieval_status == "reader_recovering"
        assert request.workspace.backend == "canonical"
        assert correction.id in {source.message_id for source in request.raw_evidence}
        assert old.id not in {source.message_id for source in request.raw_evidence}
        await service.finish(lease)

        reader.recovering = False
        await service.accept(owner, chat.id, "after-recovery", "森屿陶艺小碗的釉色是什么？")
        lease = await service.claim("recovered")
        request = await service.generation_request(lease)
        assert request.workspace.backend == "mem0_timem"
        assert {old.id, correction.id} <= {source.message_id for source in request.raw_evidence}
        await service.finish(lease)


@pytest.mark.parametrize("change", ["replacement", "selection", "preference", "forget"])
async def test_late_native_result_cannot_survive_control_change(conversation, change):
    import asyncio

    from hmph.conversation import Conversation
    from hmph.memory import Memory
    from hmph.memory_types import MemoryProposal
    from hmph.preference_types import ExpressionPreferences
    from hmph.preferences import Preferences

    identity = await conversation.bootstrap()
    await say(conversation, identity, "我叫小禾。", "name")
    memory = Memory(conversation.db, digest_key=conversation.digest_key)
    await memory.learn(
        await memory.claim("extract"),
        [MemoryProposal(predicate="name", value="小禾", quote="我叫小禾")],
    )
    fact = (await memory.entries(identity.owner_id, identity.character_id))[0]
    build, _, native = await ready_graph(conversation, identity)
    selection = MemorySelection(conversation.db, digest_key=conversation.digest_key)
    await selection.update(
        identity.owner_id, identity.character_id, "select", expected_revision=1, build_id=build.id
    )
    entered, release = asyncio.Event(), asyncio.Event()

    class PausedReader(NativeReader):
        async def retrieve_sources(self, ticket, query):
            entered.set()
            await release.wait()
            return await super().retrieve_sources(ticket, query)

    service = Conversation(
        conversation.db, digest_key=conversation.digest_key, source_reader=PausedReader(native)
    )
    chat = await service.create_chat(identity.owner_id, identity.character_id, "query")
    turn = await service.accept(identity.owner_id, chat.id, "query", "我叫什么？")
    lease = await service.claim("reader")
    task = asyncio.create_task(service.generation_request(lease))
    await asyncio.wait_for(entered.wait(), 2)
    if change == "replacement":
        await ready_graph(conversation, identity)
    elif change == "selection":
        await selection.update(
            identity.owner_id, identity.character_id, "baseline", expected_revision=2, build_id=None
        )
    elif change == "preference":
        await Preferences(conversation).update(
            identity.owner_id,
            chat.id,
            "prefs",
            expected_revision=1,
            preferences=ExpressionPreferences(nickname="小禾"),
        )
    else:
        receipt = await memory.forget(identity.owner_id, fact.id, "forget", expected_revision=1)
        assert receipt.status == "pending_cleanup"  # No wait on a live reader.
    release.set()
    with pytest.raises(DomainError, match="stale_lease"):
        await task
    with pytest.raises(DomainError, match="stale_lease"):
        await service.publish(lease, 0, "迟到的旧结果。")
    assert (await service.get_turn(identity.owner_id, turn.id)).status == "invalidated"
