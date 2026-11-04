"""Native Graphiti providers and isolated group lifecycle for local research only."""

import asyncio
import json
import os
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[2]
GRAPHITI_REVISION = "16bf514380de4e44884aed6779bece220e6fe470"
NEO4J_URI = "bolt://127.0.0.1:17687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "hmph_graphiti_local_only"


def local_embedder():
    from fastembed import TextEmbedding
    from graphiti_core.embedder.client import EmbedderClient, EmbedderConfig

    class FastEmbedClient(EmbedderClient):
        def __init__(self):
            self.config = EmbedderConfig(embedding_dim=384)
            self.model = TextEmbedding(
                model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
                cache_dir=str(ROOT / "artifacts/models/multilingual-minilm"),
                threads=4,
                providers=["CPUExecutionProvider"],
            )
            self.lock = asyncio.Lock()

        async def create(self, input_data):
            texts = [input_data] if isinstance(input_data, str) else list(input_data)
            if len(texts) != 1 or not isinstance(texts[0], str):
                raise ValueError("Graphiti local embedder expects one string or create_batch")
            return (await self.create_batch(texts))[0]

        async def create_batch(self, input_data_list):
            if not input_data_list:
                return []
            async with self.lock:
                vectors = await asyncio.to_thread(
                    lambda: list(self.model.embed(input_data_list, batch_size=32))
                )
            return [v.tolist() for v in vectors]

    return FastEmbedClient()


def valid_on(date):
    from graphiti_core.search.search_filters import ComparisonOperator, DateFilter, SearchFilters

    when = datetime.fromisoformat(date)
    when = when.replace(tzinfo=UTC) if when.tzinfo is None else when.astimezone(UTC)
    return SearchFilters(
        valid_at=[
            [DateFilter(date=when, comparison_operator=ComparisonOperator.less_than_equal)],
            [DateFilter(comparison_operator=ComparisonOperator.is_null)],
        ],
        invalid_at=[
            [DateFilter(date=when, comparison_operator=ComparisonOperator.greater_than)],
            [DateFilter(comparison_operator=ComparisonOperator.is_null)],
        ],
    )


async def graph_context(graph, group, query, date, *, max_bytes=12000):
    """Native combined RRF plus bounded provenance expansion, without new model calls."""
    from graphiti_core.nodes import EpisodicNode
    from graphiti_core.search.search_config_recipes import COMBINED_HYBRID_SEARCH_RRF

    if not 512 <= max_bytes <= 12000:
        raise ValueError("Graphiti reference reader budget must be 512..12000 bytes")
    config = COMBINED_HYBRID_SEARCH_RRF.model_copy(deep=True)
    config.limit = 8
    config.community_config = None
    config.edge_config.sim_min_score = 0
    config.node_config.sim_min_score = 0
    result = await graph.search_(
        query, config=config, group_ids=[group], search_filter=valid_on(date) if date else None
    )
    source_ids = [e.uuid for e in result.episodes]
    source_ids += [s for edge in result.edges for s in edge.episodes]
    if result.nodes and len(set(source_ids)) < 8:
        # Native node-source API has no LIMIT, so bound this provenance lookup explicitly.
        rows, _, _ = await graph.driver.execute_query(
            "MATCH (e:Episodic)-[:MENTIONS]->(n:Entity) "
            "WHERE n.uuid IN $nodes AND n.group_id=$group AND e.group_id=$group "
            "RETURN DISTINCT e.uuid AS uuid, e.valid_at AS observed "
            "ORDER BY observed DESC LIMIT 8",
            nodes=[n.uuid for n in result.nodes[:4]],
            group=group,
        )
        source_ids += [row["uuid"] for row in rows]
    source_ids = list(dict.fromkeys(source_ids))[:8]
    episodes = await EpisodicNode.get_by_uuids(graph.driver, source_ids) if source_ids else []
    if any(e.group_id != group for e in episodes):
        raise RuntimeError("Graphiti source scope mismatch")
    candidates = [
        {
            "kind": "source",
            "content": e.content,
            "observed_at": e.valid_at.isoformat(),
            "source_ids": [e.uuid],
        }
        for e in sorted(episodes, key=lambda e: (e.valid_at, e.uuid))
    ]
    candidates += [
        {
            "kind": "fact",
            "fact": e.fact,
            "valid_at": e.valid_at.isoformat() if e.valid_at else None,
            "invalid_at": e.invalid_at.isoformat() if e.invalid_at else None,
            "source_ids": e.episodes,
        }
        for e in result.edges
    ]
    selected = []
    for item in candidates:
        payload = {"question": query, "memory": [*selected, item]}
        if len(json.dumps(payload, ensure_ascii=False).encode()) <= max_bytes:
            selected.append(item)
        if len(selected) == 12:
            break
    return selected


@asynccontextmanager
async def native_graphiti(client, *, model, cleanup_audit=None, max_tokens=1024):
    os.environ["GRAPHITI_TELEMETRY_ENABLED"] = "false"
    os.environ["EMBEDDING_DIM"] = "384"
    from graphiti_core import Graphiti
    from graphiti_core.cross_encoder.openai_reranker_client import OpenAIRerankerClient
    from graphiti_core.llm_client.config import LLMConfig
    from graphiti_core.llm_client.openai_generic_client import OpenAIGenericClient
    from graphiti_core.utils.maintenance.graph_data_operations import clear_data

    config = LLMConfig(
        api_key="native-placeholder",
        model=model,
        small_model=model,
        base_url="https://api.deepseek.com",
        max_tokens=max_tokens,
    )
    llm = OpenAIGenericClient(
        config=config, client=client, max_tokens=max_tokens, structured_output_mode="json_object"
    )
    graph = Graphiti(
        NEO4J_URI,
        NEO4J_USER,
        NEO4J_PASSWORD,
        llm_client=llm,
        embedder=await asyncio.to_thread(local_embedder),
        cross_encoder=OpenAIRerankerClient(config=config, client=client),
        max_coroutines=2,
    )
    group = "hmph_graphiti_probe_" + uuid4().hex
    try:
        await graph.build_indices_and_constraints()
        yield graph, group
    finally:
        try:
            # Only retire this invocation's group, never all data in the database.
            await clear_data(graph.driver, group_ids=[group])
            rows, _, _ = await graph.driver.execute_query(
                "MATCH (n) WHERE n.group_id=$group RETURN count(n) AS total", group=group
            )
            remaining = rows[0]["total"]
            if remaining:
                raise RuntimeError("Graphiti probe group cleanup failed")
            if cleanup_audit is not None:
                cleanup_audit.update(remaining_group_nodes=remaining)
        finally:
            await graph.close()
