"""Supervised native Graphiti lifecycle; model HTTPS is replaced, stores are real."""

import asyncio
import json
import multiprocessing
import time
from dataclasses import dataclass
from importlib.util import find_spec

import httpx
import pytest
from pydantic import SecretStr

if find_spec("graphiti_core") is None:
    pytest.skip("Run in artifacts/graphiti-runtime-env", allow_module_level=True)


def native_provider(_):
    return httpx.Response(
        200,
        json={
            "choices": [
                {
                    "message": {
                        "content": json.dumps(
                            {
                                "extracted_entities": [{"name": "Mira", "entity_type_id": 0}],
                                "missed_entities": [],
                                "edges": [],
                                "entity_resolutions": [
                                    {"id": 0, "name": "Mira", "duplicate_candidate_id": -1}
                                ],
                                "summary": "The user's name is Mira.",
                                "summaries": [
                                    {"name": "Mira", "summary": "The user's name is Mira."}
                                ],
                            }
                        )
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        },
    )


@dataclass
class PausedProvider:
    entered: object
    released: object

    def __call__(self, request):
        self.entered.value = 1
        deadline = time.monotonic() + 45
        while not self.released.value:
            if time.monotonic() > deadline:
                raise RuntimeError("Controlled provider deadline")
            time.sleep(0.02)
        return native_provider(request)


def graphiti_supervisor(url, schema, entered, released, phase="during_write"):
    from hmph.database import Database
    from hmph.framework_process import GraphitiProcessBackend
    from hmph.framework_worker import FrameworkWorker
    from hmph.graphiti_backend import GraphitiBackend
    from hmph.memory_builds import MemoryBuilds

    async def run():
        database = Database(url, schema=schema)
        try:
            builds = MemoryBuilds(database, digest_key=b"test-only-request-digest-key")
            native = GraphitiBackend(
                SecretStr("test-only-key"),
                database_url=url,
                database_schema=schema,
                max_calls=8,
                delegate=httpx.MockTransport(
                    PausedProvider(entered, released)
                    if phase == "during_write"
                    else native_provider
                ),
            )

            class PausedSupervisor(GraphitiProcessBackend):
                async def pause(self):
                    entered.value = 1

                    def wait():
                        deadline = time.monotonic() + 45
                        while not released.value:
                            if time.monotonic() > deadline:
                                raise RuntimeError("Supervisor test deadline")
                            time.sleep(0.02)

                    await asyncio.to_thread(wait)

                async def build(self, lease, checkpoint):
                    if phase == "before_start":
                        from uuid import uuid4

                        import psutil

                        from hmph.memory_build_types import WriterIdentity

                        identity = WriterIdentity(
                            protocol=self.protocol,
                            host_id=self.host_id,
                            boot_time=psutil.boot_time(),
                            execution_id=uuid4(),
                        )
                        await builds.reserve_execution(lease, identity)
                        await native.prepare_execution(lease, identity)
                        await self.pause()
                    else:
                        await super().build(lease, checkpoint)
                        if phase == "after_exit":
                            await self.pause()

                async def purge(self, lease):
                    proof = await super().purge(lease)
                    if phase == "during_cleanup":
                        await self.pause()
                    return proof

            await FrameworkWorker(
                builds, PausedSupervisor(builds, native), cleanup_seconds=0.5
            ).run_once()
        finally:
            await database.close()

    asyncio.run(run())


async def test_native_graphiti_build_writes_episode_through_fenced_driver(conversation):
    from uuid import uuid4

    import psutil

    from hmph.graphiti_backend import GraphitiBackend, group_name
    from hmph.graphiti_fence import FenceStore
    from hmph.memory_build_types import WriterIdentity
    from hmph.memory_builds import MemoryBuilds

    identity = await conversation.bootstrap()
    chat = await conversation.create_chat(identity.owner_id, identity.character_id, "chat")
    turn = await conversation.accept(identity.owner_id, chat.id, "fact", "My name is Mira.")
    await conversation.cancel(identity.owner_id, turn.id, "cancel")
    builds = MemoryBuilds(conversation.db, digest_key=conversation.digest_key)
    await builds.request(identity.owner_id, identity.character_id, "build", backend="graphiti")
    lease = await builds.claim("native-test", backend="graphiti")
    reserved = WriterIdentity(
        protocol="graphiti-spawn-v1",
        host_id="a" * 64,
        boot_time=psutil.boot_time(),
        execution_id=uuid4(),
    )
    await builds.reserve_execution(lease, reserved)
    native = GraphitiBackend(
        SecretStr("test-only-graphiti-key"),
        database_url=conversation.db.url,
        database_schema=conversation.db.schema,
        max_calls=8,
        delegate=httpx.MockTransport(native_provider),
    )
    control = FenceStore(native.uri, native.user, native.password)
    try:
        await native.prepare_execution(lease, reserved)

        async def checkpoint():
            await builds.renew(lease, 30)

        usage = await native.build(lease, checkpoint)
        assert usage.processed_sources == 1 and usage.native_changes >= 2 and usage.usage_complete
    finally:
        await control.initialize()
        await control.stop(lease.space_id, reserved.execution_id)
        await control.purge(lease.space_id, group_name(lease.space_id))
        await control.close()


@pytest.mark.parametrize("max_calls", [1, 8])
async def test_supervised_native_episode_publish_then_forget_cleans_persistent_space(
    conversation, max_calls
):
    from hmph.framework_process import GraphitiProcessBackend
    from hmph.framework_worker import FrameworkWorker
    from hmph.graphiti_backend import GraphitiBackend, group_name
    from hmph.graphiti_fence import FenceStore
    from hmph.memory import Memory
    from hmph.memory_builds import MemoryBuilds
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
    fact = (await memory.entries(identity.owner_id, identity.character_id))[0]
    builds = MemoryBuilds(conversation.db, digest_key=conversation.digest_key)
    requested = await builds.request(
        identity.owner_id, identity.character_id, "build", backend="graphiti"
    )
    native = GraphitiBackend(
        SecretStr("test-only-graphiti-key"),
        database_url=conversation.db.url,
        database_schema=conversation.db.schema,
        max_calls=max_calls,
        delegate=httpx.MockTransport(native_provider),
    )
    worker = FrameworkWorker(builds, GraphitiProcessBackend(builds, native))
    try:
        assert await worker.run_once()
        result = await builds.get(identity.owner_id, requested.id)
        space = (await builds.spaces(identity.owner_id, requested.id))[0]
        if max_calls == 1:
            assert result.status == "failed" and result.error_code == "call_budget"
            assert result.attempts == 1 and not await builds.claim(
                "no-paid-repeat", backend="graphiti"
            )
        else:
            assert result.status == "ready", result
            from pathlib import Path

            from hmph.conversation import Conversation
            from hmph.deepseek import DeepSeekModel
            from hmph.graphiti_reader import GraphitiSourceReader
            from hmph.online_memory import MemorySelection
            from hmph.worker import Worker

            reader = GraphitiSourceReader(
                Path("artifacts/graphiti-runtime-env/Scripts/python.exe"),
                uri=native.uri,
                user=native.user,
                password=native.password,
            )
            try:
                await reader.warmup()
                original_pid = reader.process.pid
                from types import SimpleNamespace

                reader.process.stdin.close()  # Real IPC/process loss, before a new query.
                with pytest.raises((ConnectionError, RuntimeError, ValueError)):
                    await reader.retrieve_sources(SimpleNamespace(space_id=space.id), "probe")
                await reader.warmup()
                assert reader.process.pid != original_pid
                await MemorySelection(conversation.db, digest_key=conversation.digest_key).update(
                    identity.owner_id,
                    identity.character_id,
                    "select",
                    expected_revision=1,
                    build_id=requested.id,
                )
                service = Conversation(
                    conversation.db, digest_key=conversation.digest_key, source_reader=reader
                )
                query = await service.create_chat(identity.owner_id, identity.character_id, "query")
                accepted = await service.accept(
                    identity.owner_id, query.id, "query", "What is my name?", model_mode="deepseek"
                )
                requests = []

                def answer_provider(request):
                    body = json.loads(request.content)
                    requests.append(body)
                    if body["stream"]:
                        chunks = [
                            {"choices": [{"delta": {"content": "Your name is Mira."}}]},
                            {
                                "choices": [{"delta": {}, "finish_reason": "stop"}],
                                "usage": {
                                    "prompt_tokens": 20,
                                    "completion_tokens": 5,
                                    "total_tokens": 25,
                                },
                            },
                        ]
                        return httpx.Response(
                            200,
                            text="".join("data: " + json.dumps(c) + "\n\n" for c in chunks)
                            + "data: [DONE]\n\n",
                        )
                    return httpx.Response(
                        200,
                        json={
                            "choices": [
                                {
                                    "message": {"content": '{"allow":true,"reason":"ok"}'},
                                    "finish_reason": "stop",
                                }
                            ],
                            "usage": {
                                "prompt_tokens": 20,
                                "completion_tokens": 5,
                                "total_tokens": 25,
                            },
                        },
                    )

                model = DeepSeekModel(
                    SecretStr("test-only-answer-key"),
                    transport=httpx.MockTransport(answer_provider),
                )
                await Worker(
                    service, model, model_mode="deepseek", reviewer=model.review
                ).run_once()
                assert len(requests) == 2
                assert all(
                    "My name is Mira." in json.dumps(r, ensure_ascii=False) for r in requests
                )
                assert (
                    await service.get_turn(identity.owner_id, accepted.id)
                ).status == "completed"
                assert (await service.messages(identity.owner_id, query.id))[
                    -1
                ].content == "Your name is Mira."
                audit = await service.memory_explanation(identity.owner_id, accepted.id)
                assert audit["context"]["retrieval_status"] == "ok", audit
                assert audit["context"]["retrieval_seconds"] < 3
            finally:
                await reader.close()
        assert space.quiesced and space.stop_proof == "process_lock_and_neo4j_fence"
        assert 0 < space.usage["attempted_calls"] <= max_calls and space.usage["usage_complete"]
        receipt = await memory.forget(identity.owner_id, fact.id, "forget", expected_revision=1)
        assert receipt.status == "pending_cleanup"
        assert await worker.run_once()
        assert (await memory.deletion(identity.owner_id, receipt.id)).status == "completed"
        assert (await builds.spaces(identity.owner_id, requested.id))[0].status == "cleaned"
    finally:
        control = FenceStore(native.uri, native.user, native.password)
        try:
            for space in await builds.spaces(identity.owner_id, requested.id):
                if space.execution:
                    await control.stop(space.id, space.execution["execution_id"])
                    await control.purge(space.id, group_name(space.id))
        finally:
            await control.close()


@pytest.mark.parametrize("phase", ["before_start", "during_write", "after_exit", "during_cleanup"])
async def test_killed_graphiti_supervisor_recovers_orphan_and_completes_forget(conversation, phase):
    from hmph.framework_process import GraphitiProcessBackend
    from hmph.framework_worker import FrameworkWorker
    from hmph.graphiti_backend import GraphitiBackend, group_name
    from hmph.graphiti_fence import FenceStore
    from hmph.memory import Memory
    from hmph.memory_builds import MemoryBuilds
    from hmph.memory_types import MemoryProposal

    identity = await conversation.bootstrap()
    chat = await conversation.create_chat(identity.owner_id, identity.character_id, "chat")
    turn = await conversation.accept(identity.owner_id, chat.id, "fact", "My name is Mira.")
    await conversation.cancel(identity.owner_id, turn.id, "cancel")
    memory = Memory(conversation.db, digest_key=conversation.digest_key)
    await memory.learn(
        await memory.claim("extract"),
        [MemoryProposal(predicate="name", value="Mira", quote="My name is Mira")],
    )
    fact = (await memory.entries(identity.owner_id, identity.character_id))[0]
    builds = MemoryBuilds(conversation.db, digest_key=conversation.digest_key)
    requested = await builds.request(
        identity.owner_id, identity.character_id, "build", backend="graphiti"
    )
    context = multiprocessing.get_context("spawn")
    entered, released = context.Value("b", 0, lock=False), context.Value("b", 0, lock=False)
    supervisor = context.Process(
        target=graphiti_supervisor,
        args=(conversation.db.url, conversation.db.schema, entered, released, phase),
    )
    native = GraphitiBackend(
        SecretStr("test-only-key"),
        database_url=conversation.db.url,
        database_schema=conversation.db.schema,
        max_calls=8,
        delegate=httpx.MockTransport(native_provider),
    )
    recovery = GraphitiProcessBackend(builds, native)
    if phase == "during_cleanup":
        await FrameworkWorker(builds, recovery).run_once()
        assert (await builds.get(identity.owner_id, requested.id)).status == "ready"
        receipt = await memory.forget(identity.owner_id, fact.id, "forget", expected_revision=1)
    supervisor.start()
    try:
        async with asyncio.timeout(25):
            while not entered.value:  # noqa: ASYNC110 - real external process latch
                await asyncio.sleep(0.05)
        supervisor.kill()
        await asyncio.to_thread(supervisor.join, 5)
        assert not supervisor.is_alive()
        if phase != "during_cleanup":
            receipt = await memory.forget(identity.owner_id, fact.id, "forget", expected_revision=1)
        else:
            await asyncio.sleep(0.55)  # expired cleanup lease; never used as writer stop proof
        assert receipt.status == "pending_cleanup"
        await FrameworkWorker(builds, recovery).run_once()
        assert (await memory.deletion(identity.owner_id, receipt.id)).status == "completed"
        space = (await builds.spaces(identity.owner_id, requested.id))[0]
        assert space.status == "cleaned" and space.stop_proof == "process_lock_and_neo4j_fence"
        released.value = 1
        control = FenceStore(native.uri, native.user, native.password)
        try:
            assert await control.purge(space.id, group_name(space.id))
        finally:
            await control.close()
    finally:
        released.value = 1
        if supervisor.is_alive():
            supervisor.kill()
        await asyncio.to_thread(supervisor.join, 5)
        supervisor.close()
        await recovery.recover()
        for space in await builds.spaces(identity.owner_id, requested.id):
            if space.execution:
                control = FenceStore(native.uri, native.user, native.password)
                try:
                    await control.stop(space.id, space.execution["execution_id"])
                    await control.purge(space.id, group_name(space.id))
                finally:
                    await control.close()
