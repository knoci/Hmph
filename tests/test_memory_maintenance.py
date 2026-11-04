import asyncio

import httpx
import pytest
from sqlalchemy import func, select

from hmph.api import create_app
from hmph.config import Settings
from hmph.contracts import DomainError
from hmph.memory import Memory
from hmph.memory_builds import MemoryBuilds
from hmph.memory_maintenance import MemoryMaintenance
from hmph.models import MemoryBuild, Pair


async def setup(conversation):
    identity = await conversation.bootstrap()
    chat = await conversation.create_chat(identity.owner_id, identity.character_id, "maintenance")
    builds = MemoryBuilds(conversation.db, digest_key=conversation.digest_key)
    return identity, chat, builds, MemoryMaintenance(builds)


async def add(conversation, identity, chat, content, key):
    turn = await conversation.accept(identity.owner_id, chat.id, key, content)
    lease = await conversation.claim("maintenance-source")
    await conversation.finish(lease)
    return next(
        m for m in await conversation.messages(identity.owner_id, chat.id) if m.turn_id == turn.id
    )


async def configure(maintenance, identity, **changes):
    settings = dict(
        expected_revision=0,
        enabled=True,
        backend="mem0",
        cooldown_seconds=3600,
        minimum_sources=2,
        per_attempt_calls=3,
        additional_calls=8,
    )
    settings.update(changes)
    return await maintenance.update(
        identity.owner_id, identity.character_id, "configure", **settings
    )


async def expire_cooldown(conversation):
    async with conversation.db.sessions.begin() as session:
        pair = await session.scalar(select(Pair).with_for_update())
        pair.memory_maintenance = {
            **pair.memory_maintenance,
            "last_scheduled_at": "2000-01-01T00:00:00+00:00",
        }


async def test_disabled_by_default_and_settings_changes_never_refill_budget(conversation):
    identity, _, _, maintenance = await setup(conversation)
    initial = await maintenance.read(identity.owner_id, identity.character_id)
    assert not initial.enabled and initial.remaining_calls == 0
    assert await maintenance.schedule("mem0") == []
    with pytest.raises(DomainError, match="maintenance_budget_required"):
        await configure(maintenance, identity, additional_calls=0)
    first = await configure(maintenance, identity)
    replayed = await configure(maintenance, identity)
    assert first == replayed and first.granted_calls == 8
    updated = await maintenance.update(
        identity.owner_id,
        identity.character_id,
        "cooldown-only",
        expected_revision=first.revision,
        enabled=True,
        backend="mem0",
        cooldown_seconds=7200,
        additional_calls=0,
    )
    assert updated.remaining_calls == updated.granted_calls == 8
    assert updated.reserved_calls == 0


async def test_batch_coalescing_cooldown_and_failure_suspension(conversation):
    identity, chat, builds, maintenance = await setup(conversation)
    await configure(maintenance, identity)
    await add(conversation, identity, chat, "我喜欢乌龙茶。", "source1")
    assert await maintenance.schedule("mem0") == []
    await add(conversation, identity, chat, "我住在海城。", "source2")
    await asyncio.gather(maintenance.schedule("mem0"), maintenance.schedule("mem0"))
    async with conversation.db.sessions() as session:
        assert await session.scalar(select(func.count()).select_from(MemoryBuild)) == 1
    queued = await maintenance.read(identity.owner_id, identity.character_id)
    assert queued.remaining_calls == 8 and queued.reserved_calls == 0
    lease = await builds.claim("maintenance-writer", backend="mem0")
    assert lease.call_limit == 3 and len(lease.sources) == 2
    assert await builds.claim("another-worker", backend="mem0") is None
    with pytest.raises(DomainError, match="stale_framework_lease"):
        await builds.renew(lease.model_copy(update={"call_limit": 128}), 30)
    await builds.quiesce(lease)
    assert await builds.publish(lease)
    await add(conversation, identity, chat, "我开始学习大提琴。", "source3")
    await add(conversation, identity, chat, "老师叫安琪。", "source4")
    assert await maintenance.schedule("mem0") == []
    await expire_cooldown(conversation)
    assert len(await maintenance.schedule("mem0")) == 1
    replacement = await builds.claim("maintenance-next", backend="mem0")
    assert len(replacement.sources) == 4
    await builds.quiesce(replacement)
    await builds.fail(replacement, code="provider_failed", retryable=False)
    policy = await maintenance.read(identity.owner_id, identity.character_id)
    assert not policy.enabled and policy.suspended_reason == "build_failed"
    assert policy.remaining_calls == 2 and policy.reserved_calls == 6
    assert await maintenance.schedule("mem0") == []
    async with conversation.db.sessions() as session:
        assert (await session.scalar(select(Pair))).selected_memory_build is None


async def test_reserved_caps_survive_retry_and_restart_without_refund(conversation):
    identity, chat, builds, maintenance = await setup(conversation)
    await configure(maintenance, identity, minimum_sources=1, additional_calls=4)
    await add(conversation, identity, chat, "我叫小沐。", "budget-source")
    await maintenance.schedule("mem0")
    first = await builds.claim("budget-first", backend="mem0")
    assert first.call_limit == 3
    await builds.quiesce(first)
    await builds.fail(first, retryable=True)
    second = await builds.claim("budget-second", backend="mem0")
    assert second.call_limit == 1
    await builds.quiesce(second)
    await builds.fail(second, retryable=True)
    assert await builds.claim("budget-exhausted", backend="mem0") is None
    restarted = MemoryMaintenance(MemoryBuilds(conversation.db, digest_key=conversation.digest_key))
    policy = await restarted.read(identity.owner_id, identity.character_id)
    assert policy.remaining_calls == 0 and policy.reserved_calls == 4
    assert not policy.enabled and policy.suspended_reason == "budget_exhausted"
    assert await restarted.schedule("mem0") == []


async def test_disable_invalidates_maintenance_lease_but_leaves_manual_build_alone(conversation):
    identity, chat, builds, maintenance = await setup(conversation)
    policy = await configure(maintenance, identity, minimum_sources=1)
    await add(conversation, identity, chat, "我喜欢红茶。", "disable-source")
    await maintenance.schedule("mem0")
    lease = await builds.claim("disabled-writer", backend="mem0")
    policy = await maintenance.update(
        identity.owner_id,
        identity.character_id,
        "disable",
        expected_revision=policy.revision,
        enabled=False,
        backend="mem0",
    )
    with pytest.raises(DomainError, match="stale_framework_lease"):
        await builds.renew(lease, 30)
    reserved = policy.reserved_calls
    manual = await builds.request(
        identity.owner_id, identity.character_id, "manual", backend="mem0"
    )
    await maintenance.update(
        identity.owner_id,
        identity.character_id,
        "disable-again",
        expected_revision=policy.revision,
        enabled=False,
        backend="mem0",
    )
    assert (await builds.get(identity.owner_id, manual.id)).status == "queued"
    claimed = await builds.claim("manual-writer", backend="mem0")
    assert claimed.call_limit is None and claimed.maintenance_revision is None
    assert (
        await maintenance.read(identity.owner_id, identity.character_id)
    ).reserved_calls == reserved


async def test_forget_excludes_old_source_from_later_maintenance_snapshot(conversation):
    identity, chat, builds, maintenance = await setup(conversation)
    await configure(maintenance, identity, minimum_sources=1)
    forgotten = await add(conversation, identity, chat, "我喜欢乌龙茶。", "forget-source")
    survivor = await add(conversation, identity, chat, "我住在海城。", "keep-source")
    await maintenance.schedule("mem0")
    memory = Memory(conversation.db, digest_key=conversation.digest_key)
    receipt = await memory.forget_source(
        identity.owner_id, forgotten.id, "forget", expected_revision=forgotten.revision
    )
    assert receipt.status == "completed"
    await expire_cooldown(conversation)
    await maintenance.schedule("mem0")
    lease = await builds.claim("after-forget", backend="mem0")
    assert [source.message_id for source in lease.sources] == [survivor.id]


async def test_no_empty_or_unretainable_source_batch_is_scheduled(conversation):
    identity, chat, _, maintenance = await setup(conversation)
    await configure(maintenance, identity, minimum_sources=1)
    await add(conversation, identity, chat, "不要记住我的密码 test-only-dummy-secret。", "private")
    assert await maintenance.schedule("mem0") == []
    assert (await maintenance.read(identity.owner_id, identity.character_id)).reserved_calls == 0


async def test_maintenance_api_requires_auth_and_reports_explicit_budget(conversation):
    await setup(conversation)
    token = "maintenance-fixture-token-at-least-32"
    settings = Settings(
        _env_file=None, api_token=token, request_digest_key="x" * 32, model_mode="fake"
    )
    app = create_app(settings, conversation=conversation)
    app.state.identity = await conversation.bootstrap()
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app), base_url="http://testserver"
    ) as client:
        assert (await client.get("/api/memory-maintenance")).status_code == 401
        response = await client.put(
            "/api/memory-maintenance",
            headers={"Authorization": "Bearer " + token},
            json={
                "request_id": "api-policy",
                "expected_revision": 0,
                "enabled": True,
                "backend": "mem0",
                "additional_calls": 8,
            },
        )
        assert response.status_code == 200 and response.json()["remaining_calls"] == 8
        response = await client.get(
            "/api/memory-health", headers={"Authorization": "Bearer " + token}
        )
        assert response.json()["maintenance"]["enabled"] is True
