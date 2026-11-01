"""Opt-in native worker tests: framework-env, PostgreSQL and local embedding.

Only provider HTTPS is replaced; the lifecycle and native stores are real.
"""

import json
from importlib.util import find_spec

import httpx
import pytest
import pytest_asyncio
from pydantic import SecretStr

if find_spec("mem0") is None:
    pytest.skip("Run in artifacts/framework-env", allow_module_level=True)


def response():
    return httpx.Response(
        200,
        json={
            "choices": [
                {
                    "message": {
                        "content": json.dumps({"memory": [{"text": "User's name is Mira."}]})
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120},
        },
    )


async def seed(conversation):
    from hmph.memory import Memory
    from hmph.memory_types import MemoryProposal

    identity = await conversation.bootstrap()
    chat = await conversation.create_chat(identity.owner_id, identity.character_id, "chat")
    turn = await conversation.accept(identity.owner_id, chat.id, "name", "My name is Mira.")
    await conversation.cancel(identity.owner_id, turn.id, "cancel")
    memory = Memory(conversation.db, digest_key=conversation.digest_key)
    await memory.learn(
        await memory.claim("extract"),
        [MemoryProposal(predicate="name", value="Mira", quote="My name is Mira")],
    )
    return identity, memory, (await memory.entries(identity.owner_id, identity.character_id))[0]


@pytest_asyncio.fixture
async def native_backend(conversation):
    from sqlalchemy import select

    from hmph.mem0_backend import Mem0Backend
    from hmph.memory_build_types import CleanupLease
    from hmph.models import MemoryBuild, MemorySpace

    created = []

    def factory(endpoint, **options):
        backend = Mem0Backend(
            SecretStr("fake-worker-key"),
            database_url=conversation.db.url,
            delegate=httpx.MockTransport(endpoint),
            **options,
        )
        created.append(backend)
        return backend

    yield factory
    # Infrastructure teardown also covers failed assertions, before the test schema is dropped.
    if created:
        async with conversation.db.sessions() as session:
            spaces = (
                await session.execute(select(MemorySpace.id, MemoryBuild.pair_id).join(MemoryBuild))
            ).all()
        for space_id, pair_id in spaces:
            await created[-1].purge(
                CleanupLease(
                    space_id=space_id,
                    pair_id=pair_id,
                    backend="mem0",
                    fence=0,
                    worker_id="teardown",
                )
            )


async def test_native_worker_persists_a_shadow_build_then_verifies_full_forget(
    conversation, native_backend
):
    from hmph.framework_worker import FrameworkWorker
    from hmph.memory_builds import MemoryBuilds

    identity, memory, fact = await seed(conversation)
    builds = MemoryBuilds(conversation.db, digest_key=conversation.digest_key)
    requested = await builds.request(
        identity.owner_id, identity.character_id, "build", backend="mem0"
    )
    backend = native_backend(lambda _: response())
    await FrameworkWorker(builds, backend).run_once()
    assert (await builds.get(identity.owner_id, requested.id)).status == "ready"
    space = (await builds.spaces(identity.owner_id, requested.id))[0]
    assert space.quiesced and space.usage == {
        "attempted_calls": 1,
        "reported_tokens": 120,
        "usage_complete": True,
        "processed_sources": 1,
        "native_changes": 1,
    }
    receipt = await memory.forget(identity.owner_id, fact.id, "forget", expected_revision=1)
    assert receipt.status == "pending_cleanup"
    restarted = MemoryBuilds(conversation.db, digest_key=conversation.digest_key)
    await FrameworkWorker(restarted, backend).run_once()
    assert (await memory.deletion(identity.owner_id, receipt.id)).status == "completed"
    assert (await builds.spaces(identity.owner_id, requested.id))[0].status == "cleaned"
    assert not await builds.current(identity.owner_id, identity.character_id, backend="mem0")


async def test_exhausted_native_call_budget_does_not_repeat_paid_builds(
    conversation, native_backend
):
    from hmph.framework_worker import FrameworkWorker
    from hmph.memory_builds import MemoryBuilds

    identity, _, _ = await seed(conversation)
    chat = await conversation.create_chat(identity.owner_id, identity.character_id, "second")
    turn = await conversation.accept(identity.owner_id, chat.id, "drink", "I prefer oolong tea.")
    await conversation.cancel(identity.owner_id, turn.id, "cancel-drink")
    builds = MemoryBuilds(conversation.db, digest_key=conversation.digest_key)
    requested = await builds.request(
        identity.owner_id, identity.character_id, "build", backend="mem0"
    )
    backend = native_backend(lambda _: response(), max_calls=1)
    await FrameworkWorker(builds, backend).run_once()
    result = await builds.get(identity.owner_id, requested.id)
    assert result.status == "failed" and result.attempts == 1 and result.error_code == "call_budget"
    assert not await builds.claim("unexpected-retry", backend="mem0")
    space = (await builds.spaces(identity.owner_id, requested.id))[0]
    assert space.quiesced and space.usage["attempted_calls"] == 1
    await FrameworkWorker(builds, backend).run_once()
    assert (await builds.spaces(identity.owner_id, requested.id))[0].status == "cleaned"
    assert not await builds.current(identity.owner_id, identity.character_id, backend="mem0")


async def test_cancelling_a_native_writer_waits_for_the_thread_before_cleanup(
    conversation, native_backend
):
    import asyncio
    import threading

    from hmph.framework_worker import FrameworkWorker
    from hmph.memory_builds import MemoryBuilds

    entered, release = threading.Event(), threading.Event()

    def endpoint(_):
        entered.set()
        if not release.wait(20):
            raise RuntimeError("Test provider wait expired")
        return response()

    identity, memory, fact = await seed(conversation)
    builds = MemoryBuilds(conversation.db, digest_key=conversation.digest_key)
    requested = await builds.request(
        identity.owner_id, identity.character_id, "build", backend="mem0"
    )
    backend = native_backend(endpoint)
    work = asyncio.create_task(FrameworkWorker(builds, backend).run_once())
    try:
        assert await asyncio.to_thread(entered.wait, 20)
        receipt = await memory.forget(identity.owner_id, fact.id, "forget", expected_revision=1)
        work.cancel()
        await asyncio.sleep(0.05)
        assert not work.done()
        assert (await builds.spaces(identity.owner_id, requested.id))[0].quiesced is False
        assert not await builds.claim_cleanup("too-early", backend="mem0")
    finally:
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await work
    assert (await builds.spaces(identity.owner_id, requested.id))[0].quiesced is True
    await FrameworkWorker(builds, backend).run_once()
    assert (await memory.deletion(identity.owner_id, receipt.id)).status == "completed"
