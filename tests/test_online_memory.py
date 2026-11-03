"""Product evidence boundary: native IDs are untrusted, PostgreSQL owns text."""

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
        identity.owner_id, identity.character_id, uuid4().hex, backend="graphiti"
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
