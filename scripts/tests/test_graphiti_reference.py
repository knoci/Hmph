"""Optional real-Neo4j native contracts; no paid model calls."""

import json
from datetime import UTC, datetime
from importlib.util import find_spec

import httpx
import pytest
from openai import AsyncOpenAI
from pydantic import SecretStr

from hmph.reference_transport import AsyncReferenceTransport

if find_spec("graphiti_core") is None:
    pytest.skip("Run native Graphiti tests in artifacts/graphiti-env", allow_module_level=True)


async def seed(graph, group, *, shared=False):
    from graphiti_core.edges import EntityEdge, EpisodicEdge
    from graphiti_core.nodes import EntityNode, EpisodeType, EpisodicNode

    start = datetime(2026, 10, 1, tzinfo=UTC)
    change = datetime(2026, 10, 15, tzinfo=UTC)
    person = EntityNode(name="Mira", group_id=group, summary="Mira")
    company = EntityNode(
        name="Juniper Works" if shared else "Cedar Atelier", group_id=group, summary="An employer"
    )
    await person.generate_name_embedding(graph.embedder)
    await company.generate_name_embedding(graph.embedder)
    await person.save(graph.driver)
    await company.save(graph.driver)
    episodes = [
        EpisodicNode(
            name=str(i),
            group_id=group,
            source=EpisodeType.text,
            source_description="synthetic fixture",
            content="Mira works at Juniper Works.",
            valid_at=start if i == 0 else change,
        )
        for i in range(2)
    ]
    facts = [
        EntityEdge(
            name="WORKS_AT",
            group_id=group,
            source_node_uuid=person.uuid,
            target_node_uuid=company.uuid,
            fact="Mira works at Juniper Works.",
            created_at=datetime(2026, 11, 2, tzinfo=UTC),
            episodes=[e.uuid for e in episodes],
            valid_at=start,
        )
    ]
    if not shared:
        facts[0].fact = "Mira worked at Cedar Atelier."
        facts[0].episodes = [episodes[0].uuid]
        facts[0].invalid_at = change
        newer_company = EntityNode(name="Juniper Works", group_id=group, summary="An employer")
        await newer_company.generate_name_embedding(graph.embedder)
        await newer_company.save(graph.driver)
        facts.append(
            EntityEdge(
                name="WORKS_AT",
                group_id=group,
                source_node_uuid=person.uuid,
                target_node_uuid=newer_company.uuid,
                fact="Mira works at Juniper Works.",
                created_at=datetime(2026, 11, 2, tzinfo=UTC),
                episodes=[episodes[1].uuid],
                valid_at=change,
            )
        )
    for fact in facts:
        await fact.generate_embedding(graph.embedder)
        await fact.save(graph.driver)
    for episode in episodes:
        episode.entity_edges = [f.uuid for f in facts if episode.uuid in f.episodes]
        episode.content = " ".join(f.fact for f in facts if episode.uuid in f.episodes)
        await episode.save(graph.driver)
        for node in (person, company):
            await EpisodicEdge(
                group_id=group,
                source_node_uuid=episode.uuid,
                target_node_uuid=node.uuid,
                created_at=datetime(2026, 11, 2, tzinfo=UTC),
            ).save(graph.driver)
    return episodes, facts


@pytest.mark.asyncio
async def test_native_graphiti_filters_fact_validity_and_isolates_groups():
    from hmph.graphiti_reference import native_graphiti, valid_on

    transport = AsyncReferenceTransport(
        SecretStr("fake-private-key"),
        delegate=httpx.MockTransport(lambda _: pytest.fail("Storage contract called a model")),
    )
    audit = {}
    async with httpx.AsyncClient(transport=transport) as http:
        sdk = AsyncOpenAI(
            api_key="placeholder",
            base_url="https://api.deepseek.com",
            http_client=http,
            max_retries=0,
        )
        async with native_graphiti(sdk, model="deepseek-flash", cleanup_audit=audit) as (
            graph,
            group,
        ):
            await seed(graph, group)
            earlier = await graph.search(
                "Mira employer", group_ids=[group], search_filter=valid_on("2026-10-02")
            )
            current = await graph.search(
                "Mira employer", group_ids=[group], search_filter=valid_on("2026-11-01")
            )
            assert any("Cedar" in edge.fact for edge in earlier)
            assert not any("Juniper" in edge.fact for edge in earlier)
            assert any("Juniper" in edge.fact for edge in current)
            assert not any("Cedar" in edge.fact for edge in current)
            assert not await graph.search("Mira employer", group_ids=[group + "_other"])
    assert audit["remaining_group_nodes"] == 0 and not transport.records


@pytest.mark.asyncio
async def test_native_episode_removal_preserves_a_fact_with_another_source():
    from hmph.graphiti_reference import native_graphiti

    def summary_response(_):
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "content": '{"summaries": ['
                            '{"name":"Mira","summary":"Mira works at Juniper Works."},'
                            '{"name":"Juniper Works","summary":"Mira works here."}]}'
                        },
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120},
            },
        )

    transport = AsyncReferenceTransport(
        SecretStr("fake-private-key"), delegate=httpx.MockTransport(summary_response)
    )
    async with httpx.AsyncClient(transport=transport) as http:
        sdk = AsyncOpenAI(
            api_key="placeholder",
            base_url="https://api.deepseek.com",
            http_client=http,
            max_retries=0,
        )
        async with native_graphiti(sdk, model="deepseek-flash") as (graph, group):
            episodes, _ = await seed(graph, group, shared=True)
            await graph.remove_episode(episodes[0].uuid)
            hits = await graph.search("Mira employer", group_ids=[group])
            assert len(hits) == 1 and hits[0].episodes == [episodes[1].uuid]
            await graph.remove_episode(episodes[1].uuid)
            assert not await graph.search("Mira employer", group_ids=[group])
    assert transport.records and all(r["status"] == "completed" for r in transport.records)


@pytest.mark.asyncio
async def test_scope_retirement_clears_a_summary_left_after_native_removal_failure():
    from hmph.graphiti_reference import native_graphiti

    transport = AsyncReferenceTransport(
        SecretStr("fake-private-key"),
        delegate=httpx.MockTransport(lambda _: httpx.Response(500, json={"error": "unavailable"})),
    )
    audit = {}
    async with httpx.AsyncClient(transport=transport) as http:
        sdk = AsyncOpenAI(
            api_key="placeholder",
            base_url="https://api.deepseek.com",
            http_client=http,
            max_retries=0,
        )
        async with native_graphiti(sdk, model="deepseek-flash", cleanup_audit=audit) as (
            graph,
            group,
        ):
            from graphiti_core.nodes import EntityNode

            episodes, facts = await seed(graph, group, shared=True)
            person = await EntityNode.get_by_uuid(graph.driver, facts[0].source_node_uuid)
            person.summary = "Mira works at Juniper Works. Retired-only detail."
            await person.load_name_embedding(graph.driver)
            await person.save(graph.driver)
            episodes[0].content += " Retired-only detail."
            await episodes[0].save(graph.driver)
            # Upstream logs the rebuild error and returns normally; it keeps the old summary.
            await graph.remove_episode(episodes[0].uuid)
            saved = await EntityNode.get_by_uuid(graph.driver, person.uuid)
            assert "Retired-only detail" in saved.summary
    assert transport.records and any(r["status"] == "failed" for r in transport.records)
    assert audit["remaining_group_nodes"] == 0


@pytest.mark.asyncio
async def test_combined_context_preserves_source_text_and_the_reader_budget():
    from hmph.graphiti_reference import graph_context, native_graphiti

    transport = AsyncReferenceTransport(
        SecretStr("fake-private-key"),
        delegate=httpx.MockTransport(lambda _: pytest.fail("Retrieval called a model")),
    )
    async with httpx.AsyncClient(transport=transport) as http:
        sdk = AsyncOpenAI(
            api_key="placeholder",
            base_url="https://api.deepseek.com",
            http_client=http,
            max_retries=0,
        )
        async with native_graphiti(sdk, model="deepseek-flash") as (graph, group):
            episodes, _ = await seed(graph, group)
            context = await graph_context(graph, group, "Mira employer", "2026-11-01")
            sources = [item for item in context if item["kind"] == "source"]
            assert sources
            assert all(item["content"] in {e.content for e in episodes} for item in sources)
            assert all(item["source_ids"][0] in {e.uuid for e in episodes} for item in sources)
            assert len(context) <= 12
            assert len(json.dumps(context).encode()) < 12000
    assert not transport.records
