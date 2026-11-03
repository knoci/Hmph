"""Isolated native source retrieval. No provider key, writer fence or text IPC.

Only pinned, trusted Graphiti Cypher passes the read admission gate. READ_ACCESS
is routing, not access control; the gate explicitly rejects mutations/procedures.
This is not a sandbox for arbitrary third-party Cypher or untrusted Python.
"""

import asyncio
import contextlib
import json
import os
import re
import sys
from pathlib import Path
from uuid import uuid4

from hmph.graphiti_fence import local_uri, validate_native_query
from hmph.online_memory import NativeSources, SourceRetrievalError

IPC_BYTES = 32768


def validate_read_query(query):
    text = validate_native_query(query)
    if re.search(
        r"\b(?:CREATE|MERGE|SET|DELETE|DETACH|REMOVE|DROP|ALTER|FOREACH|LOAD)\b", text, re.I
    ):
        raise ValueError("Native reader cannot write")
    calls = re.findall(r"\bCALL\s+([\w.]+)", text, re.I)
    if any(
        name
        not in {
            "db.index.fulltext.queryNodes",
            "db.index.fulltext.queryRelationships",
            "db.index.vector.queryNodes",
            "db.index.vector.queryRelationships",
        }
        for name in calls
    ):
        raise ValueError("Native reader procedure denied")
    return text


def read_driver(uri, user, password):
    from graphiti_core.driver.neo4j_driver import Neo4jDriver
    from neo4j import READ_ACCESS, EagerResult

    class ReadDriver(Neo4jDriver):
        def __init__(self):
            super().__init__(local_uri(uri), user, password)
            self._wire, self.client = self.client, None
            self.queries = 0

        async def build_indices_and_constraints(self, delete_existing=False):
            if delete_existing:
                raise ValueError("Native reader cannot change schema")

        async def delete_all_indexes(self):
            raise ValueError("Native reader cannot change schema")

        def with_database(self, database):
            if database != "neo4j":
                raise ValueError("Native reader cannot change database")
            return self

        def session(self, database=None):
            raise ValueError("Native reader only exposes admitted execute_query")

        def transaction(self):
            raise ValueError("Native reader only exposes admitted execute_query")

        async def execute_query(self, cypher_query_, **kwargs):
            query = validate_read_query(cypher_query_)
            params = dict(kwargs.pop("params", {}) or {})
            database = kwargs.pop("database_", "neo4j")
            kwargs.pop("routing_", None)
            if database not in (None, "neo4j") or any(k.endswith("_") for k in kwargs):
                raise ValueError("Unsupported native reader option")
            params.update(kwargs)
            self.queries += 1
            async with self._wire.session(database="neo4j", default_access_mode=READ_ACCESS) as s:
                async with await s.begin_transaction(
                    timeout=1.5, metadata={"hmph_role": "reader"}
                ) as tx:
                    result = await tx.run(query, params)
                    rows = []
                    async for row in result:
                        rows.append(row)
                        if len(rows) > 80:
                            raise ValueError("Native reader result bound exceeded")
                    summary = await result.consume()
                    await tx.commit()
                    return EagerResult(rows, summary, result.keys())

        async def close(self):
            if self._init_task:
                await self._init_task
            await self._wire.close()

    return ReadDriver()


class GraphitiSourceReader:
    """One reusable child; every query failure kills only that owned child.

    Callers include creation/model warm-up in their three-second query deadline.
    There is no automatic query retry. The next turn may start a fresh child.
    """

    def __init__(self, python, *, uri, user, password):
        self.python = str(Path(python).resolve(strict=True))
        self.config = {"uri": local_uri(uri), "user": user, "password": password}
        self.process = None
        self.lock = asyncio.Lock()
        self.recovery = None
        self.closed = False

    async def close(self):
        self.closed = True
        if self.recovery and not self.recovery.done():
            self.recovery.cancel()
            await asyncio.gather(self.recovery, return_exceptions=True)
        await self._close_process()

    async def _close_process(self):
        process, self.process = self.process, None
        if process is not None:
            if process.returncode is None:
                with contextlib.suppress(ProcessLookupError):
                    process.kill()
            await process.wait()

    async def retrieve_sources(self, ticket, query):
        if self.closed:
            raise SourceRetrievalError("reader_failed")
        if self.recovery and not self.recovery.done():
            raise SourceRetrievalError("reader_recovering")
        try:
            return await self._exchange(str(ticket.space_id), query)
        except BaseException:
            # Reinitialize once without user data; never retry a failed query.
            if not self.closed:
                self.recovery = asyncio.create_task(self._recover())
            raise

    async def _recover(self):
        try:
            async with asyncio.timeout(30):
                await self._exchange(None, None)
        except Exception:
            pass

    async def warmup(self):
        async with asyncio.timeout(30):
            if self.recovery and not self.recovery.done():
                await asyncio.shield(self.recovery)
            else:
                await self._exchange(None, None)
            if self.process is None or self.process.returncode is not None:
                raise SourceRetrievalError("reader_failed")

    async def _exchange(self, space_id, query):
        async with self.lock:
            try:
                if self.process is None or self.process.returncode is not None:
                    env = {
                        k: v
                        for k, v in os.environ.items()
                        if not k.startswith(("HMPH_", "OPENAI_", "DEEPSEEK_"))
                    }
                    env["PYTHONIOENCODING"] = "utf-8"
                    env["GRAPHITI_TELEMETRY_ENABLED"] = "false"
                    env["HF_HUB_OFFLINE"] = "1"
                    self.process = await asyncio.create_subprocess_exec(
                        self.python,
                        "-m",
                        "hmph.graphiti_reader",
                        stdin=asyncio.subprocess.PIPE,
                        stdout=asyncio.subprocess.PIPE,
                        stderr=asyncio.subprocess.DEVNULL,
                        env=env,
                        limit=IPC_BYTES,
                    )
                nonce = uuid4().hex
                packet = json.dumps(
                    {
                        "nonce": nonce,
                        "config": self.config,
                        "space_id": space_id,
                        "query": query,
                    },
                    ensure_ascii=False,
                ).encode()
                if len(packet) > IPC_BYTES - 1:
                    raise ValueError("Reader IPC request too large")
                self.process.stdin.write(packet + b"\n")
                await self.process.stdin.drain()
                line = await self.process.stdout.readline()
                if not line or len(line) > IPC_BYTES:
                    raise ValueError("Reader IPC response missing or too large")
                result = json.loads(line)
                if result.pop("nonce", None) != nonce:
                    raise ValueError("Reader IPC execution mismatch")
                return NativeSources.model_validate(result)
            except BaseException:
                await self._close_process()
                raise


async def _native_graph(config):
    import httpx
    from graphiti_core import Graphiti
    from graphiti_core.cross_encoder.openai_reranker_client import OpenAIRerankerClient
    from graphiti_core.llm_client.config import LLMConfig
    from graphiti_core.llm_client.openai_generic_client import OpenAIGenericClient
    from openai import AsyncOpenAI
    from pydantic import SecretStr

    from hmph.graphiti_backend import GraphitiBackend
    from hmph.graphiti_reference import local_embedder

    # Reuse the product fixed-version/source audit; no writable backend is opened.
    GraphitiBackend(SecretStr("reader-no-provider-key"), database_url="unused", **config)
    os.environ["EMBEDDING_DIM"] = "384"

    def denied(_):
        raise RuntimeError("Product retrieval cannot call a model provider")

    wire = httpx.AsyncClient(transport=httpx.MockTransport(denied))
    client = AsyncOpenAI(api_key="reader-no-provider-key", http_client=wire, max_retries=0)
    llm = OpenAIGenericClient(config=LLMConfig(api_key="reader-no-provider-key"), client=client)
    driver = read_driver(**config)
    graph = Graphiti(
        graph_driver=driver,
        llm_client=llm,
        embedder=await asyncio.to_thread(local_embedder),
        cross_encoder=OpenAIRerankerClient(
            config=LLMConfig(api_key="reader-no-provider-key"), client=client
        ),
        max_coroutines=2,
    )
    return graph, client


async def native_sources(graph, space_id, query):
    from graphiti_core.search.search_config_recipes import COMBINED_HYBRID_SEARCH_RRF

    from hmph.graphiti_backend import group_name

    config = COMBINED_HYBRID_SEARCH_RRF.model_copy(deep=True)
    config.limit, config.community_config = 12, None
    config.edge_config.sim_min_score = config.node_config.sim_min_score = 0
    group = group_name(space_id)
    start_queries = graph.driver.queries
    result = await graph.search_(query, config=config, group_ids=[group])
    ids = [e.uuid for e in result.episodes] + [i for e in result.edges for i in e.episodes]
    requests = 1
    if result.nodes:
        rows, _, _ = await graph.driver.execute_query(
            "MATCH (e:Episodic)-[:MENTIONS]->(n:Entity) "
            "WHERE n.uuid IN $nodes AND n.group_id=$group AND e.group_id=$group "
            "RETURN DISTINCT e.uuid AS uuid ORDER BY uuid LIMIT 12",
            nodes=[n.uuid for n in result.nodes[:12]],
            group=group,
        )
        ids += [row["uuid"] for row in rows]
        requests += 1
    return NativeSources(
        source_ids=tuple(dict.fromkeys(ids))[:80],
        requests=requests,
        database_queries=graph.driver.queries - start_queries,
    )


async def _serve():
    output = sys.stdout
    graph, client, config = None, None, None
    try:
        while line := await asyncio.to_thread(sys.stdin.buffer.readline, IPC_BYTES + 1):
            if len(line) > IPC_BYTES:
                raise ValueError("Reader IPC input bound exceeded")
            packet = json.loads(line)
            # Native libraries may print diagnostics. They never share the IPC channel.
            with contextlib.redirect_stdout(sys.stderr):
                if graph is None:
                    config = packet["config"]
                    graph, client = await _native_graph(config)
                if config != packet["config"]:
                    raise ValueError("Reader configuration changed")
                sources = (
                    await native_sources(graph, packet["space_id"], packet["query"])
                    if packet["query"] is not None
                    else NativeSources(requests=0)
                )
            output.write(
                json.dumps({"nonce": packet["nonce"], **sources.model_dump(mode="json")}) + "\n"
            )
            output.flush()
            del packet, sources, line
    finally:
        if graph is not None:
            await graph.close()
        if client is not None:
            await client.close()


if __name__ == "__main__":
    try:
        asyncio.run(_serve())
    except Exception:
        # Never serialize database/provider exceptions, query text or configuration.
        sys.exit(1)
