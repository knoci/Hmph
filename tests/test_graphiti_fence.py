"""Write admission contracts; optional integration uses a dedicated local Neo4j."""

import asyncio
from importlib.util import find_spec
from uuid import uuid4

import pytest

from hmph.graphiti_fence import FenceClosed, FenceStore, fenced_driver, validate_native_query


@pytest.mark.parametrize(
    "query",
    [
        "MATCH (f:HmphWriterFence) SET f.active=true",
        "CREATE INDEX hidden FOR (n:Entity) ON (n.uuid)",
        "CALL { CREATE (n:Entity) } IN TRANSACTIONS",
        "MATCH (n) DETACH DELETE n",
        "CALL apoc.cypher.run($query,{})",
    ],
)
def test_native_writer_rejects_control_schema_and_unfenced_paths(query):
    with pytest.raises(ValueError):
        validate_native_query(query)


def test_pinned_relationship_merge_accepts_only_prebound_native_endpoints():
    query = (
        "MATCH (episode:Episodic {uuid:$episode}) MATCH (node:Entity {uuid:$entity}) "
        "MERGE (episode)-[e:MENTIONS {uuid:$edge}]->(node) SET e.group_id=$group"
    )
    assert validate_native_query(query) == query
    with pytest.raises(ValueError):
        validate_native_query("MERGE (unbound)-[e:MENTIONS]->(target) SET e.group_id=$group")
    with pytest.raises(ValueError):
        validate_native_query(
            "MATCH (source:Entity {uuid:$source}) MERGE (source)-[e:MENTIONS]->(unbound)"
        )


native = pytest.mark.skipif(
    find_spec("graphiti_core") is None, reason="Run in the isolated Graphiti runtime environment"
)
URI = "bolt://127.0.0.1:17688"
PASSWORD = "hmph_graphiti_runtime_local_only"
WRITE = "CREATE (n:Entity {uuid:$uuid, group_id:$group}) RETURN n.uuid AS uuid"


@native
async def test_product_reader_exposes_only_admitted_read_queries():
    from hmph.graphiti_reader import read_driver

    driver = read_driver(URI, "neo4j", PASSWORD)
    try:
        rows, _, _ = await driver.execute_query("RETURN $query AS value", query="safe")
        assert rows[0]["value"] == "safe"
        assert driver.client is None
        for query in (
            WRITE,
            "MATCH (e:Entity) SET e.name='changed'",
            "CALL db.create.setNodeVectorProperty($node, 'embedding', $value)",
            "CREATE INDEX hidden FOR (n:Entity) ON (n.uuid)",
        ):
            with pytest.raises(ValueError):
                await driver.execute_query(query)
        with pytest.raises(ValueError):
            driver.session()
        with pytest.raises(ValueError):
            driver.transaction()
    finally:
        await driver.close()


@native
async def test_closed_server_fence_rejects_every_native_entrypoint_and_cannot_reopen():
    space, nonce = uuid4(), uuid4()
    group = "fence_test_" + space.hex
    store = FenceStore(URI, "neo4j", PASSWORD)
    await store.initialize()
    await store.prepare(space, nonce)
    driver = fenced_driver(URI, "neo4j", PASSWORD, space_id=space, nonce=nonce)
    try:
        rows, _, _ = await driver.client.execute_query("RETURN $query AS value", query="safe")
        assert rows[0]["value"] == "safe"
        await driver.execute_query(WRITE, uuid=str(uuid4()), group=group)
        async with driver.transaction() as tx:
            await tx.run(WRITE, uuid=str(uuid4()), group=group)
        async with driver.session() as session:

            async def write(tx):
                await tx.run(WRITE, uuid=str(uuid4()), group=group)

            await session.execute_write(write)
            await session.run(WRITE, uuid=str(uuid4()), group=group)
        await store.stop(space, nonce)
        with pytest.raises(FenceClosed):
            await store.prepare(space, nonce)
        with pytest.raises(FenceClosed):
            await driver.execute_query(WRITE, uuid=str(uuid4()), group=group)
        with pytest.raises(FenceClosed):
            async with driver.transaction():
                pass
        async with driver.session() as session:
            with pytest.raises(FenceClosed):
                await session.execute_write(write)
            with pytest.raises(FenceClosed):
                await session.run(WRITE, uuid=str(uuid4()), group=group)
        assert await store.purge(space, group)
    finally:
        await store.stop(space, nonce)
        await store.purge(space, group)
        await driver.close()
        await store.close()


@native
async def test_stop_waits_for_real_inflight_transaction_and_preserves_other_space():
    first, second, nonce, other_nonce = uuid4(), uuid4(), uuid4(), uuid4()
    group, other_group = "fence_test_" + first.hex, "fence_test_" + second.hex
    store = FenceStore(URI, "neo4j", PASSWORD)
    await store.initialize()
    await store.prepare(first, nonce)
    await store.prepare(second, other_nonce)
    driver = fenced_driver(URI, "neo4j", PASSWORD, space_id=first, nonce=nonce)
    other = fenced_driver(URI, "neo4j", PASSWORD, space_id=second, nonce=other_nonce)
    closing = None
    try:
        async with driver.transaction() as tx:
            await tx.run(WRITE, uuid=str(uuid4()), group=group)
            closing = asyncio.create_task(store.stop(first, nonce))
            with pytest.raises(TimeoutError):
                await asyncio.wait_for(asyncio.shield(closing), timeout=0.15)
            # One space's stop barrier does not serialize all native writes.
            rows, _, _ = await other.execute_query(WRITE, uuid="other", group=other_group)
            assert rows[0]["uuid"] == "other"
        await asyncio.wait_for(closing, timeout=5)
        assert await store.purge(first, group)
        with pytest.raises(FenceClosed):
            await driver.execute_query(WRITE, uuid="late", group=group)
        rows, _, _ = await other.execute_query(
            "MATCH (n:Entity {group_id:$group}) RETURN n.uuid AS uuid", group=other_group
        )
        assert rows[0]["uuid"] == "other"
    finally:
        if closing is not None:
            await asyncio.gather(closing, return_exceptions=True)
        for space, execution, selected in (
            (first, nonce, group),
            (second, other_nonce, other_group),
        ):
            await store.stop(space, execution)
            await store.purge(space, selected)
        await driver.close()
        await other.close()
        await store.close()


@native
async def test_close_before_prepare_creates_permanent_tombstone_for_late_supervisor():
    space, nonce = uuid4(), uuid4()
    store = FenceStore(URI, "neo4j", PASSWORD)
    try:
        await store.initialize()
        await store.stop(space, nonce)
        with pytest.raises(FenceClosed):
            await store.prepare(space, nonce)
        assert await store.purge(space, "fence_test_" + space.hex)
    finally:
        await store.close()
