"""Persistent TiMem + mem0 indexes using the existing fenced build lifecycle."""

import json
import logging
import sys
from itertools import groupby

from hmph.framework_process import ProcessBackend, execution_lock
from hmph.mem0_backend import Mem0Backend, Mem0SourceReader
from hmph.mem0_reference import MEM0_REVISION
from hmph.memory_build_types import FrameworkBuildError
from hmph.online_memory import SourceRetrievalError
from hmph.timem_reference import (
    TIMEM_REVISION,
    LocalTiMem,
    MeteredTiMemLLM,
    TiMemSupplement,
    build_snapshot_layers,
    install_embedding,
    install_provider,
)

BACKEND = "mem0_timem"
REVISION = TIMEM_REVISION


def configure_native(root):
    # TiMem's pinned source layout contains top-level llm and storage packages.
    repo = str(root / "artifacts/references/TiMem")
    if repo not in sys.path:
        sys.path.insert(0, repo)
    from timem.utils.logging import SimpleLogger

    # The upstream console logger includes full query and vector payloads.
    SimpleLogger._safe_print = lambda *_: None
    logging.getLogger("timem").setLevel(logging.CRITICAL)


class CombinedBackend(Mem0Backend):
    backend = BACKEND
    revision = REVISION
    summarize_leaves = True

    def __init__(self, key, *, database_url, max_calls):
        super().__init__(key, database_url=database_url, max_calls=max_calls, batch_bytes=3000)

    def _batches(self, sources):
        # Keep consecutive turns from the same conversation together for down-drill.
        batch, size = [], 0
        for source in sources:
            weight = len(source.content.encode("utf-8")) + 1
            if batch and (
                size + weight > self.batch_bytes
                or batch[0].conversation_id != source.conversation_id
            ):
                yield tuple(batch)
                batch, size = [], 0
            batch.append(source)
            size += weight
        if batch:
            yield tuple(batch)

    async def build(self, lease, checkpoint):
        from hmph.eval_budget import RunBudget

        if lease.backend != BACKEND or lease.backend_revision != REVISION:
            raise FrameworkBuildError("build_failed", retryable=False)
        batches = tuple(self._batches(lease.sources))
        episodes = []
        for _, group in groupby(batches, key=lambda batch: batch[0].conversation_id):
            adjacent = tuple(group)
            episodes.extend(adjacent[i : i + 4] for i in range(0, len(adjacent), 4))
        timem_calls = len(episodes) + (len(batches) if self.summarize_leaves else 0)
        if len(batches) + timem_calls > self.max_calls:
            raise FrameworkBuildError("call_budget", retryable=False)
        try:
            await super().build(lease, checkpoint)
        except FrameworkBuildError:
            raise
        except Exception:
            raise FrameworkBuildError("provider_failed", retryable=False) from None

        _, directory = self._location(lease.space_id)
        remaining = self.max_calls - self.usage.attempted_calls
        if remaining < timem_calls:
            raise FrameworkBuildError("call_budget", retryable=False)
        budget = RunBudget(
            directory / "timem-budget.sqlite",
            max_calls=max(1, remaining),
            max_input_bytes=8 * 1024 * 1024,
            per_case_calls=max(1, remaining),
        )
        llm = MeteredTiMemLLM(self.key, budget, "case-001")
        native = LocalTiMem()
        try:
            configure_native(self.root)
            install_provider(llm)
            await native.open(directory)
            await build_snapshot_layers(
                native,
                lease.sources,
                episodes=episodes,
                checkpoint=checkpoint,
                summarize_leaves=self.summarize_leaves,
            )
            await native.index_originals(lease.sources)
            await checkpoint()
            manifest = {
                "timem_revision": REVISION,
                "mem0_revision": MEM0_REVISION,
                "source_ids": [str(source.message_id) for source in lease.sources],
                "nodes": [node.model_dump(mode="json") for node in native.nodes.values()],
                "leaves": native.leaf_map,
            }
            (directory / "timem.json").write_text(json.dumps(manifest), encoding="utf-8")
        except Exception:
            raise FrameworkBuildError(
                "call_budget" if llm.transport.budget_exhausted else "provider_failed",
                retryable=False,
            ) from None
        finally:
            if getattr(native, "client", None):
                native.client.close()
            await llm.close()
            calls = llm.transport.records
            self.usage = self.usage.model_copy(
                update={
                    "attempted_calls": self.usage.attempted_calls + len(calls),
                    "reported_tokens": self.usage.reported_tokens
                    + sum((call.get("usage") or {}).get("total_tokens", 0) for call in calls),
                    "usage_complete": self.usage.usage_complete
                    and all(c.get("usage") for c in calls),
                }
            )
        return self.usage


class CombinedProcessBackend(ProcessBackend):
    backend = BACKEND
    protocol = "mem0_timem-spawn-v1"

    async def purge(self, lease):
        with execution_lock(self.root, lease.space_id) as acquired:
            if not acquired:
                raise RuntimeError("Index still in use")
            return await super().purge(lease)


class CombinedReader:
    def __init__(self, builds, backend):
        self.builds, self.backend = builds, backend
        configure_native(backend.root)
        install_provider(None)  # Initialize the pinned import namespace without a writer.
        self.mem0 = Mem0SourceReader(
            builds.db.url, model_cache=backend.root / "artifacts/models/multilingual-minilm"
        )

    async def warmup(self):
        # Import the native stack and initialize ONNX before accepting turns.
        install_embedding(self.mem0.embedding)
        await self._warm_native()

    async def _warm_native(self):
        import asyncio

        from mem0.vector_stores.pgvector import PGVector  # noqa: F401
        from storage.vector_adapter import VectorAdapter  # noqa: F401
        from timem.models.memory import Memory  # noqa: F401
        from timem.workflows.retrieval_nodes.hierarchical_retriever import (  # noqa: F401
            HierarchicalRetriever,
        )
        from timem.workflows.retrieval_nodes.retrieval_planner import RetrievalPlanner

        await RetrievalPlanner().run({"question": "memory warmup"})
        await asyncio.to_thread(lambda: next(iter(self.mem0.embedding.embed(["memory warmup"]))))

    async def retrieve_sources(self, ticket, query):
        from hmph.models import MemoryBuild

        if ticket.backend != BACKEND or ticket.space_id is None:
            raise SourceRetrievalError("reader_failed")
        # Cleanup uses this same lock. A query never keeps a Windows SQLite
        # handle or a copy of the original text alive between turns.
        with execution_lock(self.backend.root, ticket.space_id) as acquired:
            if not acquired:
                raise SourceRetrievalError("reader_recovering")
            _, directory = self.backend._location(ticket.space_id)
            native = LocalTiMem()
            try:
                manifest = json.loads((directory / "timem.json").read_bytes())
                from timem.models.memory import Memory

                async with self.builds.db.sessions() as session:
                    build = await session.get(MemoryBuild, ticket.build_id)
                    if build is None or build.status != "ready" or build.pair_id != ticket.pair_id:
                        raise SourceRetrievalError("reader_failed")
                    if build.backend_revision != REVISION:
                        raise SourceRetrievalError("reader_failed")
                    sources = await self.builds._sources(session, build)
                if (
                    manifest["timem_revision"] != REVISION
                    or manifest["mem0_revision"] != MEM0_REVISION
                    or manifest["source_ids"] != [str(source.message_id) for source in sources]
                ):
                    raise SourceRetrievalError("reader_failed")
                await native.open(directory, embedding=self.mem0.embedding, existing=True)
                native.snapshot = sources
                nodes = [Memory.model_validate(row) for row in manifest["nodes"]]
                native.nodes = {node.id: node for node in nodes}
                native.leaf_map = manifest["leaves"]
                return await TiMemSupplement(
                    self.mem0,
                    native,
                    build_id=ticket.build_id,
                    space_id=ticket.space_id,
                    originals=True,
                ).retrieve_sources(ticket, query)
            except (OSError, KeyError, ValueError):
                mem0 = await self.mem0.retrieve_sources(ticket, query)
                return mem0.model_copy(
                    update={
                        "branch_status": {
                            "mem0": "ok" if mem0.source_ids else "empty",
                            "timem": "reader_failed",
                        }
                    }
                )
            finally:
                if getattr(native, "client", None):
                    native.client.close()
