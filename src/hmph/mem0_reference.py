"""Isolated native Mem0 lifecycle for local research, not the product memory backend.

Mem0 owns extraction, indexing, search and CRUD. This module only configures its
providers and cleans up the temporary PostgreSQL schema and SQLite history.
Run in the separate mem0 environment; imports of native dependencies are lazy.
"""

import os
from contextlib import contextmanager
from pathlib import Path
from urllib.parse import urlsplit
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[2]
MEM0_REVISION = "753496cf1435d76c3a711f3af51303f7fdcb6420"
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
PLACEHOLDER = "harmless-native-placeholder-not-a-credential"


@contextmanager
def native_mem0(directory, client, *, model, database_url=None, cleanup_audit=None):
    directory = Path(directory).resolve()
    allowed = (ROOT / "artifacts/references/runs").resolve()
    if not directory.is_relative_to(allowed) or directory == allowed or directory.exists():
        raise ValueError("Mem0 probe requires a fresh directory under artifacts/references/runs")
    database_url = database_url or os.getenv(
        "HMPH_TEST_DATABASE_URL",
        "postgresql://hmph:hmph_local_only@127.0.0.1:54329/hmph",
    )
    database_url = database_url.replace("postgresql+asyncpg://", "postgresql://", 1)
    if urlsplit(database_url).hostname not in {"127.0.0.1", "localhost", "::1"}:
        raise ValueError("Mem0 probe requires a loopback test database")
    directory.mkdir(parents=True)
    os.environ["MEM0_TELEMETRY"] = "False"
    os.environ["MEM0_DIR"] = str(directory / "native-config")
    os.environ["FASTEMBED_CACHE_PATH"] = str(ROOT / "artifacts/models/multilingual-minilm")

    import psycopg
    from mem0 import Memory
    from psycopg import sql
    from psycopg.conninfo import make_conninfo

    schema = "mem0_probe_" + uuid4().hex
    history = directory / "history.db"
    native = None
    created = False
    try:
        with psycopg.connect(database_url, autocommit=True) as admin:
            admin.execute(sql.SQL("CREATE SCHEMA {}").format(sql.Identifier(schema)))
            created = True
        connection = make_conninfo(database_url, options=f"-csearch_path={schema},public")
        native = Memory.from_config(
            {
                "llm": {
                    "provider": "deepseek",
                    "config": {"model": model, "api_key": PLACEHOLDER, "max_tokens": 1024},
                },
                "embedder": {
                    "provider": "fastembed",
                    "config": {"model": EMBEDDING_MODEL, "embedding_dims": 384},
                },
                "vector_store": {
                    "provider": "pgvector",
                    "config": {
                        "connection_string": connection,
                        # Unique even if upstream's collection discovery only inspects public.
                        "collection_name": schema,
                        "embedding_model_dims": 384,
                        "hnsw": False,
                        "diskann": False,
                        "maxconn": 3,
                    },
                },
                "history_db_path": str(history),
            }
        )
        native.llm.client.close()
        native.llm.client = client
        yield native
    finally:
        if native is not None:
            native.close()
            # Upstream close() closes SQLite, not its PG pools. Close owned pools too.
            for store in (native.vector_store, getattr(native, "_entity_store", None)):
                if store is not None:
                    store.connection_pool.close()
        if created:
            with psycopg.connect(database_url, autocommit=True) as admin:
                admin.execute(sql.SQL("DROP SCHEMA {} CASCADE").format(sql.Identifier(schema)))
                exists = admin.execute(
                    "SELECT EXISTS(SELECT 1 FROM pg_namespace WHERE nspname=%s)", (schema,)
                ).fetchone()[0]
                if exists:
                    raise RuntimeError("Mem0 probe schema cleanup failed")
        for path in (
            history,
            history.with_name("history.db-wal"),
            history.with_name("history.db-shm"),
        ):
            path.unlink(missing_ok=True)
        if cleanup_audit is not None:
            cleanup_audit.update(
                schema_removed=created, sqlite_history_removed=not history.exists()
            )
