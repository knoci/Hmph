"""Neo4j write admission and a durable, serializable stop barrier.

The pinned native driver is reused, with every execution entry point routed
through an explicit transaction. Each transaction first locks the permanent
space fence and checks its nonce. Closing that same fence therefore waits for
older transactions; later queued work observes the closed fence and rolls back.
No client-side polling or driver disconnect is used as proof of quiescence.
"""

import re
from contextlib import asynccontextmanager
from urllib.parse import urlsplit
from uuid import UUID


class FenceClosed(RuntimeError):
    pass


def local_uri(uri):
    parsed = urlsplit(uri)
    if parsed.scheme != "bolt" or parsed.hostname not in {"localhost", "127.0.0.1", "::1"}:
        raise ValueError("Graphiti requires a loopback Bolt database")
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ValueError("Graphiti credentials must be separate from the URI")
    return uri


def validate_native_query(query):
    """Accept pinned data-query paths, never DDL/control or nested transactions.

    This is an admission restriction for audited native code, not a general
    Cypher sandbox. User/model text is passed as parameters, never query text.
    """
    text = str(query)
    if re.search(
        r"HmphWriterFence|\bIN\s+(?:CONCURRENT\s+)?TRANSACTIONS\b|;|"
        r"\b(?:CREATE|DROP)\s+(?:INDEX|CONSTRAINT|DATABASE)\b|\bLOAD\s+CSV\b",
        text,
        re.I,
    ):
        raise ValueError("Unsupported native transaction path")
    procedures = re.findall(r"\bCALL\s+([a-zA-Z_][\w.]*)", text, re.I)
    allowed = {
        "db.index.fulltext.queryNodes",
        "db.index.fulltext.queryRelationships",
        "db.index.vector.queryNodes",
        "db.index.vector.queryRelationships",
        "db.create.setNodeVectorProperty",
        "db.create.setRelationshipVectorProperty",
    }
    if any(name not in allowed for name in procedures):
        raise ValueError("Unsupported native procedure")
    # The permanent control node has no native labels/uuid/group_id. Native
    # writers must not use global/unlabelled node mutation or cleanup.
    if re.search(r"\b(?:CREATE|MERGE|SET|DELETE|REMOVE)\b", text, re.I):
        for match in re.finditer(r"\b(MATCH|MERGE|CREATE)\s*\(\s*(\w+)\s*(\{|\))", text, re.I):
            operation, variable, ending = match.groups()
            # Pinned bulk edge saves use MERGE (episode)-[...] after binding
            # that endpoint with MATCH (episode:Episodic {...}). This creates
            # a relationship, not an unlabelled node or access to the fence.
            bound = re.search(
                rf"\(\s*{re.escape(variable)}\s*:\s*(?:Entity|Episodic|Community)\b",
                text[: match.start()],
                re.I,
            )
            relationship = re.match(
                r"\s*(?:<-|-)\s*\[.*?\]\s*(?:->|-)\s*\(\s*(\w+)\s*\)",
                text[match.end() :],
                re.S,
            )
            target_bound = relationship and re.search(
                rf"\(\s*{re.escape(relationship[1])}\s*:\s*(?:Entity|Episodic|Community)\b",
                text[: match.start()],
                re.I,
            )
            if (
                operation.upper() in {"MERGE", "CREATE"}
                and ending == ")"
                and bound
                and relationship
                and target_bound
            ):
                continue
            raise ValueError("Unscoped native node mutation")
    return text


LOCK = (
    "MATCH (f:HmphWriterFence {space: $space}) "
    "SET f.lock_version = coalesce(f.lock_version, 0) + 1 "
    "RETURN f.nonce AS nonce, f.active AS active"
)


class FenceStore:
    """Supervisor-only control access; writers receive only a fenced driver."""

    def __init__(self, uri, user, password):
        from neo4j import AsyncGraphDatabase

        self.driver = AsyncGraphDatabase.driver(
            local_uri(uri), auth=(user, password), max_transaction_retry_time=0
        )

    async def close(self):
        await self.driver.close()

    async def initialize(self):
        await self.driver.execute_query(
            "CREATE CONSTRAINT hmph_writer_space IF NOT EXISTS "
            "FOR (f:HmphWriterFence) REQUIRE f.space IS UNIQUE",
            database_="neo4j",
        )

    async def prepare(self, space_id, nonce):
        space, nonce = UUID(str(space_id)).hex, str(UUID(str(nonce)))
        records, _, _ = await self.driver.execute_query(
            "MERGE (f:HmphWriterFence {space: $space}) "
            "ON CREATE SET f.nonce=$nonce, f.active=true, f.lock_version=0 "
            "RETURN f.nonce AS nonce, f.active AS active",
            space=space,
            nonce=nonce,
            database_="neo4j",
        )
        if len(records) != 1 or records[0]["nonce"] != nonce or not records[0]["active"]:
            raise FenceClosed("Space execution cannot be reopened")

    async def stop(self, space_id, nonce):
        space, nonce = UUID(str(space_id)).hex, str(UUID(str(nonce)))
        async with self.driver.session(database="neo4j") as session:
            async with await session.begin_transaction(
                timeout=10,
                metadata={"hmph_space": space, "hmph_execution": nonce, "hmph_role": "fence_close"},
            ) as tx:
                # An absent node becomes a permanent closed tombstone. A late
                # prepare MERGE can never change it back to active, including a
                # supervisor killed while its initial MERGE was in flight.
                result = await tx.run(
                    "MERGE (f:HmphWriterFence {space: $space}) "
                    "ON CREATE SET f.nonce=$nonce, f.active=false, f.lock_version=0 "
                    "SET f.lock_version=coalesce(f.lock_version,0)+1 "
                    "RETURN f.nonce AS nonce",
                    space=space,
                    nonce=nonce,
                )
                record = await result.single(strict=True)
                if record["nonce"] != nonce:
                    raise FenceClosed("Space execution identity mismatch")
                await tx.run(
                    "MATCH (f:HmphWriterFence {space: $space}) SET f.active=false",
                    space=space,
                )
                await tx.commit()

    async def purge(self, space_id, group):
        """Only after a confirmed stop: no native control node has group_id."""
        space = UUID(str(space_id)).hex
        records, _, _ = await self.driver.execute_query(
            "MATCH (f:HmphWriterFence {space:$space}) RETURN f.active AS active",
            space=space,
            database_="neo4j",
        )
        if len(records) != 1 or records[0]["active"] is not False:
            raise FenceClosed("Cleanup requires a closed permanent fence")
        await self.driver.execute_query(
            "MATCH (n) WHERE n.group_id=$group DETACH DELETE n",
            group=group,
            database_="neo4j",
        )
        records, _, _ = await self.driver.execute_query(
            "MATCH (n) WHERE n.group_id=$group RETURN count(n) AS remaining",
            group=group,
            database_="neo4j",
        )
        return records[0]["remaining"] == 0


class _Transaction:
    def __init__(self, tx):
        self._tx = tx

    async def run(self, cypher_query_, **params):
        return await self._tx.run(validate_native_query(cypher_query_), params)


class _Session:
    def __init__(self, owner):
        self.owner = owner

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_):
        await self.close()

    async def close(self):
        pass

    async def execute_write(self, callback, *args, **kwargs):
        async with self.owner.transaction() as tx:
            return await callback(tx, *args, **kwargs)

    async def run(self, cypher_query_, **kwargs):
        # Legacy session.run has implicit/autocommit semantics in Neo4j; turn
        # it into one explicit guarded transaction and materialize before commit.
        from neo4j import EagerResult

        rows, summary, keys = await self.owner.execute_query(cypher_query_, **kwargs)
        return EagerResult(rows, summary, keys)


class _Client:
    """Do not expose a raw driver via the native public client attribute."""

    def __init__(self, owner):
        self.owner = owner

    def session(self, **kwargs):
        if set(kwargs) - {"database"}:
            raise ValueError("Unsupported native session options")
        return self.owner.session(kwargs.get("database"))

    async def execute_query(self, cypher_query_, **kwargs):
        return await self.owner.execute_query(cypher_query_, **kwargs)

    async def close(self):
        await self.owner._wire.close()

    async def verify_connectivity(self):
        await self.owner._wire.verify_connectivity()


def fenced_driver(uri, user, password, *, space_id, nonce):
    from graphiti_core.driver.neo4j_driver import Neo4jDriver

    class FencedNeo4jDriver(Neo4jDriver):
        def __init__(self):
            # Native operations objects remain unchanged. Its scheduled schema
            # setup resolves to our no-op; supervisor initialization owns DDL.
            super().__init__(local_uri(uri), user, password)
            self._wire = self.client
            self.client = _Client(self)
            self.space = UUID(str(space_id)).hex
            self.nonce = str(UUID(str(nonce)))

        async def build_indices_and_constraints(self, delete_existing=False):
            if delete_existing:
                raise ValueError("Native writer cannot change database schema")

        async def delete_all_indexes(self):
            raise ValueError("Native writer cannot change database schema")

        def with_database(self, database):
            if database != "neo4j":
                raise ValueError("Native writer cannot change database")
            return self

        def session(self, database=None):
            if database not in (None, "neo4j"):
                raise ValueError("Native writer cannot change database")
            return _Session(self)

        @asynccontextmanager
        async def transaction(self):
            async with self._wire.session(database="neo4j") as session:
                async with await session.begin_transaction(
                    timeout=30,
                    metadata={"hmph_space": self.space, "hmph_execution": self.nonce},
                ) as tx:
                    record = await (await tx.run(LOCK, space=self.space)).single(strict=True)
                    if record["active"] is not True or record["nonce"] != self.nonce:
                        raise FenceClosed("Space execution is closed")
                    yield _Transaction(tx)
                    await tx.commit()

        async def execute_query(self, cypher_query_, **kwargs):
            from neo4j import EagerResult

            params = dict(kwargs.pop("params", {}) or {})
            database = kwargs.pop("database_", "neo4j")
            kwargs.pop("routing_", None)
            if database not in (None, "neo4j"):
                raise ValueError("Native writer cannot change database")
            if any(key.endswith("_") for key in kwargs):
                raise ValueError("Unsupported native query option")
            params.update(kwargs)
            async with self.transaction() as tx:
                result = await tx.run(cypher_query_, **params)
                records = [record async for record in result]
                summary = await result.consume()
                return EagerResult(records, summary, result.keys())

    return FencedNeo4jDriver()
