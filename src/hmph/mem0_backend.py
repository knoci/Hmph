"""Opt-in persistent Mem0 spaces for shadow builds, using native algorithms.

Install in the separate framework environment. Each attempt owns exactly one
generated PG schema and one directory; cleanup verifies both are absent.
"""

import asyncio
import json
import os
import shutil
import subprocess
import time
from importlib.metadata import version
from pathlib import Path
from urllib.parse import urlsplit
from uuid import UUID

import httpx

from hmph.mem0_reference import EMBEDDING_MODEL, MEM0_REVISION, PLACEHOLDER
from hmph.memory_build_types import BuildUsage, CleanupProof, FrameworkBuildError
from hmph.reference_transport import ReferenceTransport


def validate_extraction(payload):
    """Validate the pinned native writer's protocol before its forgiving parser.

    Keep its JSON/code-fence parsing, but distinguish an explicit empty memory
    list from malformed output. No extraction or update algorithm is replaced.
    """
    from mem0.memory.utils import extract_json, remove_code_blocks

    choices = payload["choices"]
    if len(choices) != 1:
        raise ValueError("Invalid extraction response")
    content = remove_code_blocks(choices[0]["message"]["content"])
    try:
        data = json.loads(content, strict=False)
    except json.JSONDecodeError:
        data = json.loads(extract_json(content), strict=False)
    if not isinstance(data, dict) or not isinstance(data.get("memory"), list):
        raise ValueError("Invalid extraction response")
    if any(
        not isinstance(item, dict)
        or not isinstance(item.get("text"), str)
        or not item["text"].strip()
        for item in data["memory"]
    ):
        raise ValueError("Invalid extraction response")


class Mem0Backend:
    backend = "mem0"

    def __init__(
        self, key, *, database_url, max_calls=64, delegate=None, batch_bytes=None,
        budget=None, stop_file=None,
    ):
        self.database_url = database_url.replace("postgresql+asyncpg://", "postgresql://", 1)
        if urlsplit(self.database_url).hostname not in {"127.0.0.1", "localhost", "::1"}:
            raise ValueError("Local framework worker requires a loopback database")
        if not 1 <= max_calls <= 128:
            raise ValueError("Invalid framework call budget")
        if any(
            version(package) != expected
            for package, expected in (
                ("mem0ai", "2.2.1"),
                ("psycopg", "3.3.6"),
                ("openai", "2.3.0"),
                ("fastembed", "0.9.1"),
            )
        ):
            raise RuntimeError("Install the pinned Mem0 version in the framework environment")
        self.root = Path.cwd().resolve()
        repo = self.root / "artifacts/references/mem0"
        revision = subprocess.check_output(
            ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
        ).strip()
        if (
            revision != MEM0_REVISION
            or subprocess.run(["git", "-C", str(repo), "diff", "HEAD", "--quiet"]).returncode
        ):
            raise RuntimeError("Native Mem0 checkout must match the unmodified pinned revision")
        self.key, self.max_calls, self.delegate = key, max_calls, delegate
        if batch_bytes is not None and not 1 <= batch_bytes <= 3000:
            raise ValueError("Invalid Mem0 batch size")
        self.batch_bytes = batch_bytes
        self.budget, self.stop_file = budget, stop_file
        self.usage = BuildUsage(
            attempted_calls=0,
            reported_tokens=0,
            usage_complete=True,
            processed_sources=0,
            native_changes=0,
        )

    def _location(self, space_id):
        identifier = UUID(str(space_id)).hex
        base = (self.root / "artifacts/framework-memory").resolve()
        directory = (base / identifier).resolve()
        if directory.parent != base or not directory.is_relative_to(self.root):
            raise ValueError("Framework space path escaped the workspace")
        return "hmph_m0_" + identifier, directory

    def _open(self, lease, transport):
        os.environ["MEM0_TELEMETRY"] = "False"
        schema, directory = self._location(lease.space_id)
        directory.mkdir(parents=True, exist_ok=False)
        os.environ["MEM0_DIR"] = str(directory / "native-config")
        os.environ["FASTEMBED_CACHE_PATH"] = str(self.root / "artifacts/models/multilingual-minilm")
        import psycopg
        from mem0 import Memory
        from openai import OpenAI
        from psycopg import sql
        from psycopg.conninfo import make_conninfo

        writer_connection = make_conninfo(self.database_url, application_name=schema)
        with psycopg.connect(writer_connection, autocommit=True, connect_timeout=5) as admin:
            admin.execute(sql.SQL("CREATE SCHEMA {}").format(sql.Identifier(schema)))
        connection = make_conninfo(writer_connection, options=f"-csearch_path={schema},public")
        http = httpx.Client(
            transport=transport, timeout=30, follow_redirects=False, trust_env=False
        )
        try:
            native = Memory.from_config(
                {
                    "llm": {
                        "provider": "deepseek",
                        "config": {
                            "model": lease.model,
                            "api_key": PLACEHOLDER,
                            "max_tokens": 1024,
                        },
                    },
                    "embedder": {
                        "provider": "fastembed",
                        "config": {
                            "model": EMBEDDING_MODEL,
                            "embedding_dims": 384,
                        },
                    },
                    "vector_store": {
                        "provider": "pgvector",
                        "config": {
                            "connection_string": connection,
                            "collection_name": schema,
                            "embedding_model_dims": 384,
                            "hnsw": False,
                            "diskann": False,
                            "maxconn": 3,
                        },
                    },
                    "history_db_path": str(directory / "history.db"),
                }
            )
        except BaseException:
            http.close()
            raise
        native.llm.client.close()
        native.llm.client = OpenAI(
            api_key=PLACEHOLDER,
            base_url="https://api.deepseek.com",
            http_client=http,
            max_retries=0,
        )
        return native

    @staticmethod
    def _close(native):
        try:
            native.close()
        finally:
            try:
                for store in (native.vector_store, getattr(native, "_entity_store", None)):
                    if store is not None:
                        store.connection_pool.close()
            finally:
                native.llm.client.close()

    def _batches(self, sources):
        if self.batch_bytes is None:
            yield from ((source,) for source in sources)
            return
        batch, size = [], 0
        for source in sources:
            source_size = len(source.content.encode("utf-8"))
            if batch and (
                source.observed_at != batch[0].observed_at
                or size + source_size + 1 > self.batch_bytes
            ):
                yield tuple(batch)
                batch, size = [], 0
            size += source_size + bool(batch)
            batch.append(source)
        if batch:
            yield tuple(batch)

    async def build(self, lease, checkpoint):
        self.usage = BuildUsage(
            attempted_calls=0,
            reported_tokens=0,
            usage_complete=True,
            processed_sources=0,
            native_changes=0,
        )
        if lease.backend_revision != MEM0_REVISION or lease.backend != self.backend:
            raise FrameworkBuildError("build_failed", retryable=False)
        transport = ReferenceTransport(
            self.key,
            model=lease.model,
            max_calls=min(self.max_calls, lease.call_limit or self.max_calls),
            delegate=self.delegate,
            response_validator=validate_extraction,
            budget=self.budget,
            stop_file=self.stop_file,
        )
        native = None
        processed, changes = 0, 0
        try:
            batches = tuple(self._batches(lease.sources))
            if len(batches) > min(self.max_calls, lease.call_limit or self.max_calls):
                raise FrameworkBuildError("call_budget", retryable=False)
            await checkpoint()
            native = await asyncio.to_thread(self._open, lease, transport)
            for batch in batches:
                await checkpoint()
                source = batch[0]
                metadata = {
                    "hmph_source_id": str(source.message_id),
                    "hmph_source_revision": source.revision,
                    "hmph_source_order": source.source_order,
                    "hmph_observed_at": source.observed_at.isoformat(),
                }
                if len(batch) > 1:
                    metadata["hmph_source_ids"] = [str(item.message_id) for item in batch]
                result = await asyncio.to_thread(
                    native.add,
                    [{"role": "user", "content": "\n".join(item.content for item in batch)}],
                    user_id="pair_" + lease.pair_id.hex,
                    metadata=metadata,
                )
                processed += len(batch)
                changes += len(result.get("results", []))
            if any(r["status"] != "completed" for r in transport.records):
                raise RuntimeError("Framework provider failed")
            if transport.budget_exhausted:
                raise FrameworkBuildError("call_budget", retryable=False)
        except Exception:
            if transport.budget_exhausted:
                raise FrameworkBuildError("call_budget", retryable=False) from None
            if any(r.get("error_code") == "reference_invalid_output" for r in transport.records):
                raise FrameworkBuildError("provider_failed", retryable=False) from None
            raise
        finally:
            try:
                if native is not None:
                    await asyncio.to_thread(self._close, native)
            finally:
                self.usage = BuildUsage(
                    attempted_calls=len(transport.records),
                    reported_tokens=sum(
                        (r.get("usage") or {}).get("total_tokens", 0) for r in transport.records
                    ),
                    usage_complete=all(r.get("usage") is not None for r in transport.records),
                    processed_sources=processed,
                    native_changes=changes,
                )
                transport.close()
        return self.usage

    def _purge(self, lease):
        import psycopg
        from psycopg import sql

        schema, directory = self._location(lease.space_id)
        with psycopg.connect(
            self.database_url,
            autocommit=True,
            connect_timeout=5,
            options="-clock_timeout=5000 -cstatement_timeout=30000",
        ) as admin:
            admin.execute(
                sql.SQL("DROP SCHEMA IF EXISTS {} CASCADE").format(sql.Identifier(schema))
            )
            remaining = admin.execute(
                "SELECT EXISTS(SELECT 1 FROM pg_namespace WHERE nspname=%s)", (schema,)
            ).fetchone()[0]
        # _location resolved and checked the absolute target before recursive removal.
        if directory.exists():
            shutil.rmtree(directory)
        return CleanupProof(store_empty=not remaining, auxiliary_empty=not directory.exists())

    async def purge(self, lease):
        if lease.backend != self.backend:
            raise ValueError("Framework cleanup backend mismatch")
        return await asyncio.to_thread(self._purge, lease)

    def _drain_writes(self, lease):
        import psycopg

        application, _ = self._location(lease.space_id)
        with psycopg.connect(
            self.database_url,
            autocommit=True,
            connect_timeout=5,
            options="-cstatement_timeout=5000",
        ) as admin:
            until = time.monotonic() + 5
            while True:
                active = admin.execute(
                    "SELECT pid FROM pg_stat_activity "
                    "WHERE application_name=%s AND pid<>pg_backend_pid()",
                    (application,),
                ).fetchall()
                if not active:
                    return
                admin.execute(
                    "SELECT pg_terminate_backend(pid) FROM pg_stat_activity "
                    "WHERE application_name=%s AND pid<>pg_backend_pid()",
                    (application,),
                ).fetchall()
                if time.monotonic() >= until:
                    raise RuntimeError("Framework database writes did not stop")
                time.sleep(0.02)

    async def drain_writes(self, lease):
        if lease.backend != self.backend:
            raise ValueError("Framework backend mismatch")
        await asyncio.to_thread(self._drain_writes, lease)

class Mem0SourceReader:
    """Read the existing Mem0 vector collection without opening its write stores."""

    def __init__(self, database_url, *, model_cache):
        from fastembed import TextEmbedding

        self.database_url = database_url.replace("postgresql+asyncpg://", "postgresql://", 1)
        if urlsplit(self.database_url).hostname not in {"127.0.0.1", "localhost", "::1"}:
            raise ValueError("Local Mem0 reader requires a loopback database")
        self.embedding = TextEmbedding(
            model_name=EMBEDDING_MODEL, cache_dir=str(model_cache), local_files_only=True
        )

    async def retrieve_sources(self, ticket, query):
        from hmph.online_memory import CANDIDATES, NativeSources, SourceRetrievalError

        if ticket.backend != "mem0" or ticket.space_id is None:
            raise SourceRetrievalError("reader_failed")
        rows = await asyncio.to_thread(self._search, ticket.space_id, ticket.pair_id, query)
        sources = {}
        for row in rows:
            values = row.payload.get("hmph_source_ids")
            if values is None:
                values = (row.payload["hmph_source_id"],)
            if not isinstance(values, (list, tuple)) or not values:
                raise SourceRetrievalError("reader_failed")
            for value in values:
                sources[UUID(value)] = None
                if len(sources) == CANDIDATES:
                    break
            if len(sources) == CANDIDATES:
                break
        return NativeSources(source_ids=tuple(sources), requests=1, database_queries=1)

    def _search(self, space_id, pair_id, query):
        import psycopg
        from mem0.vector_stores.pgvector import PGVector

        schema = "hmph_m0_" + space_id.hex
        # This native provider normally lazily creates missing tables. In a ready
        # space a missing table is an error, and the DB enforces read-only access.
        vector = PGVector(
            dbname=None, collection_name=schema, embedding_model_dims=384,
            user=None, password=None, host=None, port=None, diskann=False, hnsw=False,
            connection_pool=None,
            connection_string=psycopg.conninfo.make_conninfo(
                self.database_url,
                options=f"-csearch_path={schema},public -cdefault_transaction_read_only=on "
                        "-cstatement_timeout=2000",
                connect_timeout=2,
            ),
            minconn=0, maxconn=1,
        )
        vector._collection_ensured = True
        try:
            embedding = next(iter(self.embedding.embed([query.replace("\n", " ")]))).tolist()
            return vector.search(
                query, embedding, top_k=12, filters={"user_id": "pair_" + pair_id.hex}
            )
        finally:
            vector.connection_pool.close()
