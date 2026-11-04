"""Opt-in Graphiti shadow spaces. Native algorithms, durable fenced writes.

Use a separate runtime environment/database, not the disposable P1 probe. API
activation and online retrieval are intentionally separate integration tasks.
"""

import asyncio
import os
import subprocess
from importlib.metadata import version
from pathlib import Path
from uuid import UUID
from zoneinfo import ZoneInfo

import httpx

from hmph.graphiti_fence import FenceClosed, FenceStore, fenced_driver, local_uri
from hmph.graphiti_reference import GRAPHITI_REVISION, local_embedder
from hmph.memory_build_types import BuildUsage, CleanupProof, FrameworkBuildError
from hmph.reference_transport import AsyncReferenceTransport

RUNTIME_URI = "bolt://127.0.0.1:17688"
RUNTIME_USER = "neo4j"
RUNTIME_PASSWORD = "hmph_graphiti_runtime_local_only"


def group_name(space_id):
    return "hmph_graphiti_space_" + UUID(str(space_id)).hex


def source_body(source):
    observed = source.observed_at
    timezone = "local timezone unknown; do not infer relative local dates"
    if source.source_timezone is not None:
        observed = observed.astimezone(ZoneInfo(source.source_timezone))
        timezone = source.source_timezone
    return (
        f"Statement observed at {observed.isoformat()} [{timezone}] "
        "(observation time, not necessarily the event time):\n" + source.content
    )


def empty_usage():
    return BuildUsage(
        attempted_calls=0,
        reported_tokens=0,
        usage_complete=True,
        processed_sources=0,
        native_changes=0,
    )


class GraphitiBackend:
    backend = "graphiti"

    def __init__(
        self,
        key,
        *,
        database_url,
        database_schema="public",
        uri=RUNTIME_URI,
        user=RUNTIME_USER,
        password=RUNTIME_PASSWORD,
        max_calls=64,
        delegate=None,
        call_budget=None,
        stop_file=None,
    ):
        if not 1 <= max_calls <= 128:
            raise ValueError("Invalid framework call budget")
        if any(
            version(package) != expected
            for package, expected in (
                ("graphiti-core", "0.31.2"),
                ("neo4j", "6.4.1"),
                ("openai", "2.3.0"),
                ("fastembed", "0.9.1"),
            )
        ):
            raise RuntimeError("Install the pinned Graphiti runtime environment")
        self.root = Path.cwd().resolve()
        repo = self.root / "artifacts/references/Graphiti"
        revision = subprocess.check_output(
            ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
        ).strip()
        if (
            revision != GRAPHITI_REVISION
            or subprocess.run(["git", "-C", str(repo), "diff", "HEAD", "--quiet"]).returncode
        ):
            raise RuntimeError("Native Graphiti checkout must match its pinned revision")
        self.uri, self.user, self.password = local_uri(uri), user, password
        self.key, self.max_calls, self.delegate = key, max_calls, delegate
        self.call_budget = call_budget
        self.stop_file = stop_file
        self.database_url, self.database_schema = database_url, database_schema
        self.execution_id = None
        self.usage = empty_usage()

    def _control(self):
        return FenceStore(self.uri, self.user, self.password)

    async def prepare_execution(self, lease, identity):
        """Supervisor only. No native writer is started until setup completes."""
        if lease.backend != self.backend or identity.protocol != "graphiti-spawn-v1":
            raise ValueError("Graphiti execution backend mismatch")
        from graphiti_core.driver.neo4j_driver import Neo4jDriver

        # Index setup has no personal data writes and belongs to the supervisor,
        # never to the fenced writer's implicit startup task.
        initializer = Neo4jDriver(self.uri, self.user, self.password)
        try:
            await asyncio.wait_for(initializer._init_task, timeout=30)
        finally:
            await initializer.close()
        control = self._control()
        try:
            await control.initialize()
            await control.prepare(lease.space_id, identity.execution_id)
        finally:
            await control.close()
        self.execution_id = identity.execution_id

    async def _mapping(self, lease, episode_ids):
        from hmph.database import Database
        from hmph.memory_builds import MemoryBuilds
        from hmph.models import MemoryNativeSource

        database = Database(self.database_url, schema=self.database_schema)
        try:
            await MemoryBuilds(
                database, digest_key=b"unused-by-native-mapping"
            ).register_native_sources(
                lease,
                [
                    MemoryNativeSource(
                        space_id=lease.space_id,
                        native_id=native_id,
                        message_id=source.message_id,
                        message_revision=source.revision,
                        source_order=source.source_order,
                    )
                    for source, native_id in zip(lease.sources, episode_ids, strict=True)
                ],
            )
        finally:
            await database.close()

    async def _open(self, client, lease):
        from graphiti_core import Graphiti
        from graphiti_core.cross_encoder.openai_reranker_client import OpenAIRerankerClient
        from graphiti_core.llm_client.config import LLMConfig
        from graphiti_core.llm_client.openai_generic_client import OpenAIGenericClient

        config = LLMConfig(
            api_key="native-placeholder",
            model=lease.model,
            small_model=lease.model,
            base_url="https://api.deepseek.com",
            max_tokens=1024,
        )
        graph_driver = fenced_driver(
            self.uri,
            self.user,
            self.password,
            space_id=lease.space_id,
            nonce=self.execution_id,
        )
        try:
            return Graphiti(
                graph_driver=graph_driver,
                llm_client=OpenAIGenericClient(
                    config=config,
                    client=client,
                    max_tokens=1024,
                    structured_output_mode="json_object",
                ),
                embedder=await asyncio.to_thread(local_embedder),
                cross_encoder=OpenAIRerankerClient(config=config, client=client),
                max_coroutines=2,
            )
        except BaseException:
            await graph_driver.close()
            raise

    async def build(self, lease, checkpoint):
        from graphiti_core.nodes import EpisodeType
        from openai import AsyncOpenAI

        if (
            lease.backend != self.backend
            or lease.backend_revision != GRAPHITI_REVISION
            or self.execution_id is None
        ):
            raise FrameworkBuildError("build_failed", retryable=False)
        os.environ["GRAPHITI_TELEMETRY_ENABLED"] = "false"
        os.environ["EMBEDDING_DIM"] = "384"
        self.usage = empty_usage()
        transport = AsyncReferenceTransport(
            self.key,
            model=lease.model,
            max_calls=min(self.max_calls, lease.call_limit or self.max_calls),
            delegate=self.delegate,
            budget=self.call_budget,
            stop_file=self.stop_file,
        )
        processed, changes = 0, 0
        graph = None
        try:
            await checkpoint()
            async with httpx.AsyncClient(
                transport=transport, timeout=30, follow_redirects=False, trust_env=False
            ) as http:
                client = AsyncOpenAI(
                    api_key="native-placeholder",
                    base_url="https://api.deepseek.com",
                    http_client=http,
                    max_retries=0,
                )
                graph = await self._open(client, lease)
                episode_ids = []
                for source in lease.sources:
                    await checkpoint()
                    result = await graph.add_episode(
                        name="source_" + str(source.source_order),
                        episode_body=source_body(source),
                        source_description="user conversation source",
                        source=EpisodeType.text,
                        reference_time=source.observed_at,
                        group_id=group_name(lease.space_id),
                        update_communities=False,
                    )
                    if (
                        result.episode.group_id != group_name(lease.space_id)
                        or result.episode.content != source_body(source)
                        or result.episode.uuid in episode_ids
                    ):
                        raise FrameworkBuildError("build_failed", retryable=False)
                    UUID(result.episode.uuid)
                    episode_ids.append(result.episode.uuid)
                    processed += 1
                    changes += 1 + len(result.nodes) + len(result.edges)
                rows, _, _ = await graph.driver.execute_query(
                    "MATCH (e:Episodic) WHERE e.group_id=$group RETURN e.uuid AS uuid",
                    group=group_name(lease.space_id),
                )
                if {row["uuid"] for row in rows} != set(episode_ids):
                    raise FrameworkBuildError("build_failed", retryable=False)
                # In this pinned version uuid means "load an existing episode".
                # Keep native creation unchanged and bind actual returned UUIDs
                # before publication, rather than inventing native identifiers.
                await self._mapping(lease, episode_ids)
                if any(record["status"] != "completed" for record in transport.records):
                    raise FrameworkBuildError("provider_failed", retryable=False)
        except Exception as error:
            if self.stop_file is not None and self.stop_file.exists():
                raise FrameworkBuildError("provider_failed", retryable=False) from None
            if transport.budget_exhausted:
                raise FrameworkBuildError("call_budget", retryable=False) from None
            if isinstance(error, (ValueError, TypeError, FenceClosed)):
                raise FrameworkBuildError("build_failed", retryable=False) from None
            raise
        finally:
            try:
                if graph is not None:
                    await graph.close()
            finally:
                self.usage = BuildUsage(
                    attempted_calls=len(transport.records),
                    reported_tokens=sum(
                        (record.get("usage") or {}).get("total_tokens", 0)
                        for record in transport.records
                    ),
                    usage_complete=all(
                        record.get("usage") is not None for record in transport.records
                    ),
                    processed_sources=processed,
                    native_changes=changes,
                )
                await transport.aclose()
        return self.usage

    async def drain_writes(self, lease, identity):
        if lease.backend != self.backend or identity.protocol != "graphiti-spawn-v1":
            raise ValueError("Graphiti execution backend mismatch")
        control = self._control()
        try:
            await control.initialize()
            await control.stop(lease.space_id, identity.execution_id)
        finally:
            await control.close()

    async def purge(self, lease):
        if lease.backend != self.backend:
            raise ValueError("Graphiti cleanup backend mismatch")
        control = self._control()
        try:
            empty = await control.purge(lease.space_id, group_name(lease.space_id))
            # Graphiti has no per-space text files in this adapter. Source maps
            # are removed by finish_cleanup only after this physical-store proof.
            return CleanupProof(store_empty=empty, auxiliary_empty=True)
        finally:
            await control.close()
