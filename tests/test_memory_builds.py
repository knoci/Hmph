from hmph.model_gateway import FakeModel
from hmph.worker import Worker


def test_graphiti_source_observation_uses_the_original_local_calendar():
    from datetime import UTC, datetime
    from uuid import uuid4

    from hmph.graphiti_backend import source_body
    from hmph.memory_build_types import BuildSource

    source = BuildSource(
        message_id=uuid4(),
        revision=1,
        source_order=1,
        content="我明天去面试。",
        observed_at=datetime(2026, 11, 2, 17, tzinfo=UTC),
        source_timezone="Asia/Hong_Kong",
    )
    assert "2026-11-03T01:00:00+08:00" in source_body(source)
    assert "Asia/Hong_Kong" in source_body(source)
    assert "local timezone unknown" in source_body(
        source.model_copy(update={"source_timezone": None})
    )


async def test_retired_supervised_attempt_cannot_register_a_late_child(conversation):
    from uuid import uuid4

    import pytest

    from hmph.contracts import DomainError
    from hmph.memory_build_types import WriterIdentity
    from hmph.memory_builds import MemoryBuilds

    identity = await conversation.bootstrap()
    await say(conversation, identity, "我叫小沐。", "name")
    builds = MemoryBuilds(conversation.db, digest_key=conversation.digest_key)
    await builds.request(identity.owner_id, identity.character_id, "old", backend="mem0")
    lease = await builds.claim("old-supervisor", backend="mem0")
    reserved = WriterIdentity(host_id="a" * 64, boot_time=100, execution_id=uuid4())
    await builds.reserve_execution(lease, reserved)
    with pytest.raises(DomainError, match="framework_stop_proof_required"):
        await builds.quiesce(lease)
    await builds.request(identity.owner_id, identity.character_id, "new", backend="mem0")
    await builds.confirm_execution_stopped(lease, reserved)
    with pytest.raises(DomainError, match="stale_framework_lease"):
        await builds.start_execution(
            lease, reserved, reserved.model_copy(update={"pid": 1234, "started_at": 110})
        )
    assert not await builds.publish(lease)
    new = await builds.claim("new-supervisor", backend="mem0")
    assert new.space_id != lease.space_id


async def say(conversation, identity, text, request_id):
    chat = await conversation.create_chat(identity.owner_id, identity.character_id, request_id)
    turn = await conversation.accept(identity.owner_id, chat.id, request_id, text)
    await conversation.cancel(identity.owner_id, turn.id, request_id)
    return (await conversation.messages(identity.owner_id, chat.id))[0]


async def test_unchanged_ready_snapshot_reuses_build_but_explicit_force_rebuilds(conversation):
    from sqlalchemy import func, select

    from hmph.memory_builds import MemoryBuilds
    from hmph.models import MemoryBuild, MemorySpace

    identity = await conversation.bootstrap()
    await say(conversation, identity, "我喜欢乌龙茶。", "reuse-source")
    builds = MemoryBuilds(conversation.db, digest_key=conversation.digest_key)
    first = await builds.request(
        identity.owner_id, identity.character_id, "first-ready", backend="mem0"
    )
    lease = await builds.claim("reuse-only-test", backend="mem0")
    await builds.quiesce(lease)
    assert await builds.publish(lease)
    reused = await builds.request(
        identity.owner_id, identity.character_id, "reuse-ready", backend="mem0"
    )
    assert reused.id == first.id and reused.status == "ready"
    async with conversation.db.sessions() as session:
        assert await session.scalar(select(func.count()).select_from(MemoryBuild)) == 1
        assert await session.scalar(select(func.count()).select_from(MemorySpace)) == 1
    forced = await builds.request(
        identity.owner_id, identity.character_id, "forced-ready", backend="mem0", force=True
    )
    assert forced.id != first.id and forced.status == "queued"


async def test_memory_health_reports_real_watermark_gap_without_building(conversation):
    from uuid import uuid4

    import pytest
    from sqlalchemy import select

    from hmph.contracts import DomainError
    from hmph.memory_builds import MemoryBuilds
    from hmph.models import MemoryBuild, MemorySpace, Pair

    identity = await conversation.bootstrap()
    builds = MemoryBuilds(conversation.db, digest_key=conversation.digest_key)
    await say(conversation, identity, "我喜欢乌龙茶。", "health-first")
    status = await builds.health(identity.owner_id, identity.character_id)
    assert status["mode"] == "baseline" and status["latest_build_status"] is None
    assert status["pending_extractions"] == 1
    frozen = await builds.request(
        identity.owner_id, identity.character_id, "health-build", backend="graphiti"
    )
    assert (await builds.health(identity.owner_id, identity.character_id))[
        "latest_build_status"
    ] == "queued"
    # Simulated lifecycle metadata tests advisory display, not Graphiti readiness.
    async with conversation.db.sessions.begin() as session:
        build = await session.get(MemoryBuild, frozen.id)
        build.status = "ready"
        session.add(
            MemorySpace(
                build_id=build.id,
                status="ready",
                fence=build.fence,
                worker_id="health-fixture",
                quiesced=True,
            )
        )
        pair = await session.scalar(select(Pair).where(Pair.owner_id == identity.owner_id))
        pair.selected_memory_build = build.id
    await say(conversation, identity, "最近换成喝红茶。", "health-second")
    status = await builds.health(identity.owner_id, identity.character_id)
    assert status["mode"] == "experimental_index" and status["new_source_count"] == 1
    assert status["build_watermark"] == frozen.watermark
    assert status["automatic_framework_maintenance"] is False
    async with conversation.db.sessions.begin() as session:
        (await session.get(MemoryBuild, frozen.id)).status = "retired"
    status = await builds.health(identity.owner_id, identity.character_id)
    assert status["mode"] == "fallback" and status["new_source_count"] is None
    with pytest.raises(DomainError, match="not_found"):
        await builds.health(uuid4(), identity.character_id)


async def test_changed_sources_do_not_reuse_old_ready_snapshot(conversation):
    from hmph.memory_builds import MemoryBuilds

    identity = await conversation.bootstrap()
    await say(conversation, identity, "我喜欢乌龙茶。", "old-reuse-source")
    builds = MemoryBuilds(conversation.db, digest_key=conversation.digest_key)
    first = await builds.request(
        identity.owner_id, identity.character_id, "old-ready", backend="mem0"
    )
    lease = await builds.claim("changed-reuse-test", backend="mem0")
    await builds.quiesce(lease)
    assert await builds.publish(lease)
    await say(conversation, identity, "我现在改喝红茶。", "new-reuse-source")
    updated = await builds.request(
        identity.owner_id, identity.character_id, "changed-ready", backend="mem0"
    )
    assert updated.id != first.id and updated.source_count == 2


async def test_build_freezes_retained_user_sources_and_survives_service_restart(conversation):
    from hmph.memory_builds import MemoryBuilds

    identity = await conversation.bootstrap()
    first = await say(conversation, identity, "我叫小沐。", "name")
    await say(conversation, identity, "不要记住这句话。", "private")
    chat = await conversation.create_chat(identity.owner_id, identity.character_id, "reply")
    await conversation.accept(identity.owner_id, chat.id, "reply", "我喜欢乌龙茶。")
    await Worker(conversation, FakeModel()).run_once()
    await say(conversation, identity, "sk-fake-source-credential-1234567890", "credential")
    builds = MemoryBuilds(conversation.db, digest_key=conversation.digest_key)
    requested = await builds.request(
        identity.owner_id, identity.character_id, "build", backend="mem0"
    )
    await say(conversation, identity, "后来我搬家了。", "later")

    restarted = MemoryBuilds(conversation.db, digest_key=conversation.digest_key)
    assert (
        await restarted.request(identity.owner_id, identity.character_id, "build", backend="mem0")
    ).id == requested.id
    lease = await restarted.claim("builder", backend="mem0")
    assert lease.build_id == requested.id
    assert [s.content for s in lease.sources] == ["我叫小沐。", "我喜欢乌龙茶。"]
    assert lease.sources[0].message_id == first.id
    assert lease.sources[0].revision == 1 and lease.sources[0].source_order == 1
    assert lease.sources[0].observed_at.tzinfo is not None
    assert not await restarted.current(identity.owner_id, identity.character_id, backend="mem0")
    await restarted.quiesce(lease)
    assert await restarted.publish(lease)
    ready = await restarted.current(identity.owner_id, identity.character_id, backend="mem0")
    assert ready.id == requested.id and ready.source_count == 2 and ready.watermark == 4
    assert ready.status == "ready" and ready.mode == "shadow"


async def test_expired_writer_cannot_publish_and_retry_owns_a_fresh_space(conversation):
    import asyncio

    from hmph.memory_builds import MemoryBuilds

    identity = await conversation.bootstrap()
    await say(conversation, identity, "我喜欢乌龙茶。", "tea")
    builds = MemoryBuilds(conversation.db, digest_key=conversation.digest_key)
    requested = await builds.request(
        identity.owner_id, identity.character_id, "build", backend="mem0"
    )
    old = await builds.claim("lost", backend="mem0", lease_seconds=0.01)
    await asyncio.sleep(0.02)
    await builds.recover_expired(backend="mem0")
    claims = await asyncio.gather(
        *(builds.claim(worker, backend="mem0") for worker in ("retry-a", "retry-b"))
    )
    active = [lease for lease in claims if lease is not None]
    assert len(active) == 1
    new = active[0]
    assert new.space_id != old.space_id and new.fence > old.fence
    spaces = await builds.spaces(identity.owner_id, requested.id)
    assert [(s.status, s.quiesced) for s in spaces] == [("retired", False), ("writing", False)]
    await builds.quiesce(old)
    assert not await builds.publish(old)
    await builds.renew(new, 30)
    await builds.quiesce(new)
    assert await builds.publish(new)
    ready = await builds.current(identity.owner_id, identity.character_id, backend="mem0")
    assert ready.attempts == 2
    assert ready.error_code is None


async def test_forget_waits_for_writer_stop_and_verified_cross_store_cleanup(conversation):
    import pytest

    from hmph.contracts import DomainError
    from hmph.memory import Memory
    from hmph.memory_build_types import CleanupProof
    from hmph.memory_builds import MemoryBuilds
    from hmph.memory_types import MemoryProposal

    identity = await conversation.bootstrap()
    await say(conversation, identity, "我叫小沐。", "name")
    memory = Memory(conversation.db, digest_key=conversation.digest_key)
    await memory.learn(
        await memory.claim("extract"),
        [MemoryProposal(predicate="name", value="小沐", quote="我叫小沐")],
    )
    fact = (await memory.entries(identity.owner_id, identity.character_id))[0]
    builds = MemoryBuilds(conversation.db, digest_key=conversation.digest_key)
    requested = await builds.request(
        identity.owner_id, identity.character_id, "build", backend="mem0"
    )
    writer = await builds.claim("slow", backend="mem0")
    assert (await memory.preview_forget(identity.owner_id, fact.id)).framework_spaces == 1
    receipt = await memory.forget(identity.owner_id, fact.id, "forget", expected_revision=1)
    assert receipt.status == "pending_cleanup" and receipt.completed_at is None
    assert receipt.framework_spaces == 1
    assert (await builds.get(identity.owner_id, requested.id)).status == "retired"
    assert not await builds.claim_cleanup("cleaner", backend="mem0")
    assert not await builds.publish(writer)

    await builds.quiesce(writer)
    cleanup = await builds.claim_cleanup("cleaner", backend="mem0")
    assert cleanup.space_id == writer.space_id
    with pytest.raises(DomainError, match="framework_cleanup_unverified"):
        await builds.finish_cleanup(cleanup, CleanupProof(store_empty=True, auxiliary_empty=False))
    assert (await memory.deletion(identity.owner_id, receipt.id)).status == "pending_cleanup"
    await builds.fail_cleanup(cleanup, retry_seconds=0)
    retried = await builds.claim_cleanup("retry-cleaner", backend="mem0")
    assert retried.fence > cleanup.fence
    assert not await builds.finish_cleanup(
        cleanup, CleanupProof(store_empty=True, auxiliary_empty=True)
    )
    assert await builds.finish_cleanup(
        retried, CleanupProof(store_empty=True, auxiliary_empty=True)
    )
    completed = await memory.deletion(identity.owner_id, receipt.id)
    assert completed.status == "completed" and completed.completed_at is not None
    assert (
        await memory.forget(identity.owner_id, fact.id, "forget", expected_revision=1)
    ).status == "completed"
    rebuilt = await builds.request(
        identity.owner_id, identity.character_id, "after-forget", backend="mem0"
    )
    assert rebuilt.source_count == 0 and rebuilt.memory_epoch > requested.memory_epoch


async def test_new_request_supersedes_inflight_build_and_owner_cannot_read_another_pair(
    conversation,
):
    from uuid import uuid4

    import pytest

    from hmph.contracts import DomainError
    from hmph.memory_builds import MemoryBuilds

    identity = await conversation.bootstrap()
    other = await conversation.bootstrap(uuid4())
    await conversation.create_chat(other.owner_id, other.character_id, "other")
    await say(conversation, identity, "我喜欢茶。", "tea")
    builds = MemoryBuilds(conversation.db, digest_key=conversation.digest_key)
    old = await builds.request(identity.owner_id, identity.character_id, "old", backend="mem0")
    inflight = await builds.claim("old", backend="mem0")
    new = await builds.request(identity.owner_id, identity.character_id, "new", backend="mem0")
    assert (await builds.get(identity.owner_id, old.id)).status == "retired"
    await builds.quiesce(inflight)
    assert not await builds.publish(inflight)
    with pytest.raises(DomainError, match="not_found"):
        await builds.get(other.owner_id, new.id)
    with pytest.raises(DomainError, match="not_found"):
        await builds.spaces(other.owner_id, new.id)
    with pytest.raises(DomainError, match="idempotency_conflict"):
        await builds.request(identity.owner_id, identity.character_id, "new", backend="graphiti")
    assert not await builds.current(other.owner_id, other.character_id, backend="mem0")
    latest = await builds.claim("new", backend="mem0")
    with pytest.raises(DomainError, match="framework_writer_not_quiesced"):
        await builds.publish(latest)
    await builds.quiesce(latest)
    assert await builds.publish(latest)
    replacement = await builds.request(
        identity.owner_id, identity.character_id, "replace", backend="mem0", force=True
    )
    assert (
        await builds.current(identity.owner_id, identity.character_id, backend="mem0")
    ).id == new.id
    lease = await builds.claim("replace", backend="mem0")
    await builds.quiesce(lease)
    assert await builds.publish(lease)
    assert (
        await builds.current(identity.owner_id, identity.character_id, backend="mem0")
    ).id == replacement.id
    assert (await builds.get(identity.owner_id, new.id)).status == "retired"


async def test_build_failures_stop_retrying_after_three_attempts_and_keep_usage(conversation):
    from hmph.memory_build_types import BuildUsage
    from hmph.memory_builds import MemoryBuilds

    identity = await conversation.bootstrap()
    await say(conversation, identity, "我叫小沐。", "name")
    builds = MemoryBuilds(conversation.db, digest_key=conversation.digest_key)
    build = await builds.request(identity.owner_id, identity.character_id, "build", backend="mem0")
    for index in range(3):
        lease = await builds.claim(f"attempt-{index}", backend="mem0")
        await builds.quiesce(
            lease,
            usage=BuildUsage(
                attempted_calls=1,
                reported_tokens=120,
                usage_complete=True,
                processed_sources=0,
                native_changes=0,
            ),
        )
        assert await builds.fail(lease, code="provider_failed", retryable=True)
    assert not await builds.claim("fourth", backend="mem0")
    result = await builds.get(identity.owner_id, build.id)
    assert (
        result.status == "failed"
        and result.attempts == 3
        and result.error_code == "provider_failed"
    )
    assert len(await builds.spaces(identity.owner_id, build.id)) == 3
    assert (await builds.spaces(identity.owner_id, build.id))[0].usage["reported_tokens"] == 120


async def test_cleanup_timeout_can_retry_without_accepting_the_expired_confirmation(conversation):
    import asyncio

    from hmph.memory_build_types import CleanupProof
    from hmph.memory_builds import MemoryBuilds

    identity = await conversation.bootstrap()
    await say(conversation, identity, "我喜欢茶。", "tea")
    builds = MemoryBuilds(conversation.db, digest_key=conversation.digest_key)
    await builds.request(identity.owner_id, identity.character_id, "old", backend="mem0")
    writer = await builds.claim("old-writer", backend="mem0")
    await builds.quiesce(writer)
    assert await builds.publish(writer)
    await builds.request(identity.owner_id, identity.character_id, "new", backend="mem0")
    latest = await builds.claim("new-writer", backend="mem0")
    await builds.quiesce(latest)
    assert await builds.publish(latest)
    expired = await builds.claim_cleanup("lost-cleaner", backend="mem0", lease_seconds=0.01)
    await asyncio.sleep(0.02)
    await builds.recover_expired(backend="mem0")
    cleanup = await builds.claim_cleanup("replacement", backend="mem0")
    proof = CleanupProof(store_empty=True, auxiliary_empty=True)
    assert cleanup.space_id == writer.space_id and cleanup.fence > expired.fence
    assert not await builds.finish_cleanup(expired, proof)
    assert await builds.finish_cleanup(cleanup, proof)
    assert (
        await builds.current(identity.owner_id, identity.character_id, backend="mem0")
    ).id == latest.build_id
