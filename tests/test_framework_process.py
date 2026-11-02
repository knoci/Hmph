"""Real spawned writers, PostgreSQL and local embedding; only model HTTP is fake."""

import asyncio
import json
import multiprocessing
import time
from dataclasses import dataclass
from importlib.util import find_spec

import httpx
import pytest
from pydantic import SecretStr

if find_spec("mem0") is None:
    pytest.skip("Run in artifacts/framework-env", allow_module_level=True)

from test_framework_native import native_backend, seed  # noqa: E402, F401, F811


def provider_response(_):
    return httpx.Response(
        200,
        json={
            "choices": [
                {
                    "message": {"content": json.dumps({"memory": [{"text": "Mira"}]})},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120},
        },
    )


async def test_spawned_writer_publishes_only_after_exit_and_verified_cleanup(
    conversation,
    native_backend,  # noqa: F811
):
    from hmph.framework_process import ProcessBackend
    from hmph.framework_worker import FrameworkWorker
    from hmph.memory_builds import MemoryBuilds

    identity, memory, fact = await seed(conversation)
    builds = MemoryBuilds(conversation.db, digest_key=conversation.digest_key)
    requested = await builds.request(
        identity.owner_id, identity.character_id, "build", backend="mem0"
    )
    backend = ProcessBackend(builds, native_backend(provider_response))
    await FrameworkWorker(builds, backend).run_once()
    assert (await builds.get(identity.owner_id, requested.id)).status == "ready"
    space = (await builds.spaces(identity.owner_id, requested.id))[0]
    assert space.execution["pid"] != __import__("os").getpid()
    assert space.stop_proof == "process_lock_and_database_barrier"
    assert space.usage["reported_tokens"] == 120
    receipt = await memory.forget(identity.owner_id, fact.id, "forget", expected_revision=1)
    await FrameworkWorker(builds, backend).run_once()
    assert (await memory.deletion(identity.owner_id, receipt.id)).status == "completed"


async def test_supervision_keeps_the_call_budget_failure_nonretryable(
    conversation,
    native_backend,  # noqa: F811
):
    from hmph.framework_process import ProcessBackend
    from hmph.framework_worker import FrameworkWorker
    from hmph.memory_builds import MemoryBuilds

    identity, _, _ = await seed(conversation)
    chat = await conversation.create_chat(identity.owner_id, identity.character_id, "second")
    turn = await conversation.accept(identity.owner_id, chat.id, "tea", "I prefer oolong tea.")
    await conversation.cancel(identity.owner_id, turn.id, "cancel-tea")
    builds = MemoryBuilds(conversation.db, digest_key=conversation.digest_key)
    requested = await builds.request(
        identity.owner_id, identity.character_id, "build", backend="mem0"
    )
    backend = ProcessBackend(builds, native_backend(provider_response, max_calls=1))
    await FrameworkWorker(builds, backend).run_once()
    result = await builds.get(identity.owner_id, requested.id)
    assert result.status == "failed" and result.error_code == "call_budget" and result.attempts == 1
    assert not await builds.claim("paid-retry", backend="mem0")
    space = (await builds.spaces(identity.owner_id, requested.id))[0]
    assert space.quiesced and space.usage["attempted_calls"] == 1
    await FrameworkWorker(builds, backend).run_once()
    assert (await builds.spaces(identity.owner_id, requested.id))[0].status == "cleaned"


def wait_release(release):
    # A killed Event.wait() waiter can corrupt multiprocessing.Condition bookkeeping.
    # This fault-injection latch has no cross-process lock to strand on hard kill.
    until = time.monotonic() + 60
    while not release.value:
        if time.monotonic() >= until:
            raise RuntimeError("Test deadline")
        time.sleep(0.02)


@dataclass
class ControlledProvider:
    entered: object
    release: object
    phase: str

    def __call__(self, request):
        if self.phase == "during_write":
            self.entered.set()
            wait_release(self.release)
        return provider_response(request)


def supervisor_main(url, schema, phase, entered, release):
    from hmph.database import Database
    from hmph.framework_process import ProcessBackend
    from hmph.framework_worker import FrameworkWorker
    from hmph.mem0_backend import Mem0Backend
    from hmph.memory_builds import MemoryBuilds

    async def run():
        database = Database(url, schema=schema)
        try:
            builds = MemoryBuilds(database, digest_key=b"test-only-request-digest-key")
            native = Mem0Backend(
                SecretStr("fake-process-test-key"),
                database_url=url,
                delegate=httpx.MockTransport(ControlledProvider(entered, release, phase)),
            )

            class PausedSupervisor(ProcessBackend):
                async def build(self, lease, checkpoint):
                    if phase == "before_start":
                        # Reserve without launching: crash between durable reservation and spawn.
                        from uuid import uuid4

                        import psutil

                        from hmph.memory_build_types import WriterIdentity

                        await builds.reserve_execution(
                            lease,
                            WriterIdentity(
                                host_id=self.host_id,
                                boot_time=psutil.boot_time(),
                                execution_id=uuid4(),
                            ),
                        )
                    else:
                        await super().build(lease, checkpoint)
                    entered.set()
                    await asyncio.to_thread(wait_release, release)

                async def purge(self, lease):
                    proof = await super().purge(lease)
                    if phase == "during_cleanup":
                        entered.set()
                        await asyncio.to_thread(wait_release, release)
                    return proof

            await FrameworkWorker(
                builds, PausedSupervisor(builds, native), cleanup_seconds=0.5
            ).run_once()
        finally:
            await database.close()

    asyncio.run(run())


async def wait_until(check, *, seconds=20):
    async with asyncio.timeout(seconds):
        while not (result := await check()):  # noqa: ASYNC110 - external process state
            await asyncio.sleep(0.05)
        return result


@pytest.mark.parametrize("phase", ["before_start", "during_write", "after_exit"])
async def test_supervisor_crash_is_recovered_without_resurrecting_forgotten_sources(
    conversation,
    native_backend,  # noqa: F811
    phase,  # noqa: F811
):
    from hmph.framework_process import ProcessBackend
    from hmph.framework_worker import FrameworkWorker
    from hmph.memory_builds import MemoryBuilds

    identity, memory, fact = await seed(conversation)
    builds = MemoryBuilds(conversation.db, digest_key=conversation.digest_key)
    requested = await builds.request(
        identity.owner_id, identity.character_id, "crash", backend="mem0"
    )
    context = multiprocessing.get_context("spawn")
    entered, release = context.Event(), context.Value("b", 0, lock=False)
    process = context.Process(
        target=supervisor_main,
        args=(
            conversation.db.url,
            conversation.db.schema,
            phase,
            entered,
            release,
        ),
    )
    process.start()
    try:
        assert await asyncio.to_thread(entered.wait, 25)
        space = (await builds.spaces(identity.owner_id, requested.id))[0]
        assert not space.quiesced
        process.kill()
        await asyncio.to_thread(process.join, 5)
        assert not process.is_alive()
        receipt = await memory.forget(identity.owner_id, fact.id, "forget", expected_revision=1)
        assert receipt.status == "pending_cleanup"
        restarted = ProcessBackend(builds, native_backend(provider_response))
        await FrameworkWorker(builds, restarted).run_once()
        finished = await memory.deletion(identity.owner_id, receipt.id)
        assert finished.status == "completed", (
            await builds.spaces(identity.owner_id, requested.id)
        )[0].model_dump_json()
        space = (await builds.spaces(identity.owner_id, requested.id))[0]
        assert space.quiesced and space.status == "cleaned"
        assert space.stop_proof == "process_lock_and_database_barrier"
        assert space.usage is None  # A crash must not turn unknown paid usage into zero.
        fresh = await builds.request(
            identity.owner_id, identity.character_id, "new-space", backend="mem0"
        )
        assert fresh.source_count == 0
        await FrameworkWorker(builds, restarted).run_once()
        assert (await builds.get(identity.owner_id, fresh.id)).status == "ready"
        assert (await builds.spaces(identity.owner_id, fresh.id))[0].id != space.id
    finally:
        release.value = 1
        if process.is_alive():
            process.kill()
        await asyncio.to_thread(process.join, 5)
        # Reap any orphan native child even when an assertion above failed.
        await memory.forget(identity.owner_id, fact.id, "forget", expected_revision=1)
        await ProcessBackend(builds, native_backend(provider_response)).recover()
        process.close()


async def test_cleanup_crash_after_purge_retries_without_harming_the_next_space(
    conversation,
    native_backend,  # noqa: F811
):
    from hmph.framework_process import ProcessBackend
    from hmph.framework_worker import FrameworkWorker
    from hmph.memory_builds import MemoryBuilds

    identity, memory, fact = await seed(conversation)
    builds = MemoryBuilds(conversation.db, digest_key=conversation.digest_key)
    requested = await builds.request(
        identity.owner_id, identity.character_id, "build", backend="mem0"
    )
    backend = ProcessBackend(builds, native_backend(provider_response))
    await FrameworkWorker(builds, backend).run_once()
    receipt = await memory.forget(identity.owner_id, fact.id, "forget", expected_revision=1)
    context = multiprocessing.get_context("spawn")
    entered, release = context.Event(), context.Value("b", 0, lock=False)
    process = context.Process(
        target=supervisor_main,
        args=(
            conversation.db.url,
            conversation.db.schema,
            "during_cleanup",
            entered,
            release,
        ),
    )
    process.start()
    try:
        assert await asyncio.to_thread(entered.wait, 20)
        process.kill()
        await asyncio.to_thread(process.join, 5)

        async def completed():
            await FrameworkWorker(builds, backend).run_once()
            return (await memory.deletion(identity.owner_id, receipt.id)).status == "completed"

        await wait_until(completed)
        space = (await builds.spaces(identity.owner_id, requested.id))[0]
        assert space.status == "cleaned" and space.cleanup_attempts == 2
    finally:
        release.value = 1
        if process.is_alive():
            process.kill()
        await asyncio.to_thread(process.join, 5)
        process.close()


@pytest.mark.parametrize("identity_kind", ["legacy", "foreign_host", "reused_pid"])
async def test_recovery_preserves_unknown_writers_and_never_signals_a_reused_pid(
    conversation,
    native_backend,  # noqa: F811
    identity_kind,  # noqa: F811
):
    import subprocess
    import sys
    from uuid import uuid4

    import psutil

    from hmph.framework_process import ProcessBackend
    from hmph.framework_worker import FrameworkWorker
    from hmph.memory_build_types import WriterIdentity
    from hmph.memory_builds import MemoryBuilds

    identity, memory, fact = await seed(conversation)
    builds = MemoryBuilds(conversation.db, digest_key=conversation.digest_key)
    requested = await builds.request(
        identity.owner_id, identity.character_id, "build", backend="mem0"
    )
    lease = await builds.claim("old-writer", backend="mem0")
    backend = ProcessBackend(builds, native_backend(provider_response))
    unrelated = await asyncio.to_thread(
        subprocess.Popen,
        [sys._base_executable, "-c", "import time; time.sleep(60)"],
        creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0,
    )
    try:
        if identity_kind != "legacy":
            planned = WriterIdentity(
                host_id=backend.host_id if identity_kind == "reused_pid" else "f" * 64,
                boot_time=psutil.boot_time(),
                execution_id=uuid4(),
            )
            await builds.reserve_execution(lease, planned)
            await builds.start_execution(
                lease,
                planned,
                planned.model_copy(
                    update={
                        "pid": unrelated.pid,
                        "started_at": psutil.Process(unrelated.pid).create_time()
                        + (1 if identity_kind == "reused_pid" else 0),
                    }
                ),
            )
        receipt = await memory.forget(identity.owner_id, fact.id, "forget", expected_revision=1)
        await FrameworkWorker(builds, backend).run_once()
        assert unrelated.poll() is None
        space = (await builds.spaces(identity.owner_id, requested.id))[0]
        expected = "completed" if identity_kind == "reused_pid" else "pending_cleanup"
        assert (await memory.deletion(identity.owner_id, receipt.id)).status == expected
        assert space.quiesced is (identity_kind == "reused_pid")
    finally:
        unrelated.kill()
        unrelated.wait(timeout=5)


async def test_recovery_cancels_inflight_database_writes_before_confirming_stop(
    conversation,
    native_backend,  # noqa: F811
):
    from uuid import uuid4

    import psutil
    import psycopg
    from sqlalchemy import text

    from hmph.framework_process import ProcessBackend
    from hmph.framework_worker import FrameworkWorker
    from hmph.memory_build_types import WriterIdentity
    from hmph.memory_builds import MemoryBuilds

    identity, memory, fact = await seed(conversation)
    builds = MemoryBuilds(conversation.db, digest_key=conversation.digest_key)
    requested = await builds.request(
        identity.owner_id, identity.character_id, "build", backend="mem0"
    )
    lease = await builds.claim("dead-writer", backend="mem0")
    backend = ProcessBackend(builds, native_backend(provider_response))
    await builds.reserve_execution(
        lease,
        WriterIdentity(
            host_id=backend.host_id,
            boot_time=psutil.boot_time(),
            execution_id=uuid4(),
        ),
    )
    application = "hmph_m0_" + lease.space_id.hex
    url = conversation.db.url.replace("postgresql+asyncpg://", "postgresql://", 1)

    def inflight():
        try:
            with psycopg.connect(url, application_name=application) as connection:
                connection.execute("SELECT pg_sleep(60)")
        except psycopg.OperationalError:
            return "cancelled"
        return "unexpected completion"

    writing = asyncio.create_task(asyncio.to_thread(inflight))

    async def active():
        async with conversation.db.sessions() as session:
            return await session.scalar(
                text(
                    "SELECT count(*) FROM pg_stat_activity "
                    "WHERE application_name=:app AND state='active'"
                ),
                {"app": application},
            )

    try:
        await wait_until(active)
        receipt = await memory.forget(identity.owner_id, fact.id, "forget", expected_revision=1)
        # A connection outside the owned application namespace must remain usable.
        with psycopg.connect(url, application_name="hmph-unrelated-test") as unrelated:
            await FrameworkWorker(builds, backend).run_once()
            assert unrelated.execute("SELECT 1").fetchone()[0] == 1
        assert await asyncio.wait_for(writing, 5) == "cancelled"
        assert (await memory.deletion(identity.owner_id, receipt.id)).status == "completed"
        assert (await builds.spaces(identity.owner_id, requested.id))[0].quiesced
    finally:
        await backend.recover()
        if not writing.done():
            await asyncio.wait_for(writing, 6)
