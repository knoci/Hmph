"""Single-file SQLite memory store: versioned items, bi-temporal graph, FTS5, vectors, and DAG forget."""

from __future__ import annotations

import json
import math
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from hmph.lexical import TOKENIZER_VERSION, tokenize
from hmph.local_memory.embeddings import (
    Embedder,
    HashingEmbedder,
    batch_cosine_top_k,
    build_normalized_matrix,
    cosine_similarity,
    matrix_cosine_top_k,
    pack_vector,
    unpack_vector,
)
from hmph.local_memory.reranker import extract_anchored_search_tokens, parse_anchor_date
from hmph.local_memory.types import (
    AmbiguityRecord,
    ConsolidationRecord,
    CoreMemoryBlock,
    EdgeRecord,
    EdgeRelation,
    ForgetReceipt,
    ItemKind,
    ItemRecord,
    ItemStatus,
    MessageRecord,
    SessionSummaryRecord,
    VersionRecord,
)

SCHEMA_SQL = """
PRAGMA journal_mode = WAL;
PRAGMA foreign_keys = ON;
PRAGMA synchronous = NORMAL;

CREATE TABLE IF NOT EXISTS messages (
    id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    role TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
    content TEXT NOT NULL,
    created_at TEXT NOT NULL,
    timezone TEXT NOT NULL DEFAULT 'UTC',
    extracted INTEGER NOT NULL DEFAULT 0,
    deleted INTEGER NOT NULL DEFAULT 0,
    reply_to_message_id TEXT,
    extract_attempts INTEGER NOT NULL DEFAULT 0,
    extraction_failure TEXT
);

CREATE TABLE IF NOT EXISTS memory_metadata (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS message_ingest_order (
    sequence INTEGER PRIMARY KEY AUTOINCREMENT,
    message_id TEXT NOT NULL UNIQUE REFERENCES messages(id) ON DELETE CASCADE
);

CREATE TRIGGER IF NOT EXISTS messages_assign_ingest_order AFTER INSERT ON messages
BEGIN
    INSERT INTO message_ingest_order (message_id) VALUES (NEW.id);
END;

CREATE INDEX IF NOT EXISTS idx_messages_session_time
    ON messages (session_id, created_at);
CREATE INDEX IF NOT EXISTS idx_messages_pending
    ON messages (role, extracted, deleted);

CREATE TABLE IF NOT EXISTS items (
    id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    kind TEXT NOT NULL,
    status TEXT NOT NULL,
    current_version_id TEXT,
    entities_json TEXT NOT NULL DEFAULT '[]',
    last_mentioned_at TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_items_status_updated
    ON items (status, updated_at DESC);

CREATE TABLE IF NOT EXISTS item_versions (
    id TEXT PRIMARY KEY,
    item_id TEXT NOT NULL REFERENCES items(id) ON DELETE CASCADE,
    version_no INTEGER NOT NULL,
    summary TEXT NOT NULL,
    status TEXT NOT NULL,
    effective_time_text TEXT,
    valid_from TEXT,
    valid_to TEXT,
    condition_text TEXT,
    change_type TEXT NOT NULL,
    supersedes_version_id TEXT,
    source_message_id TEXT NOT NULL REFERENCES messages(id) ON DELETE CASCADE,
    source_quote TEXT NOT NULL,
    created_at TEXT NOT NULL,
    UNIQUE (item_id, version_no)
);

CREATE INDEX IF NOT EXISTS idx_versions_item ON item_versions (item_id, version_no);
CREATE INDEX IF NOT EXISTS idx_versions_source ON item_versions (source_message_id);

CREATE TABLE IF NOT EXISTS memory_edges (
    id TEXT PRIMARY KEY,
    src_type TEXT NOT NULL,
    src_id TEXT NOT NULL,
    dst_type TEXT NOT NULL,
    dst_id TEXT NOT NULL,
    relation TEXT NOT NULL,
    source_message_id TEXT,
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_edges_src ON memory_edges (src_type, src_id);
CREATE INDEX IF NOT EXISTS idx_edges_dst ON memory_edges (dst_type, dst_id);
CREATE INDEX IF NOT EXISTS idx_edges_source_msg ON memory_edges (source_message_id);

CREATE TABLE IF NOT EXISTS ambiguities (
    id TEXT PRIMARY KEY,
    source_message_id TEXT NOT NULL REFERENCES messages(id) ON DELETE CASCADE,
    source_quote TEXT NOT NULL,
    candidate_item_ids_json TEXT NOT NULL DEFAULT '[]',
    note TEXT NOT NULL,
    resolved INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_ambiguities_open ON ambiguities (resolved, created_at DESC);

CREATE TABLE IF NOT EXISTS embeddings (
    target_type TEXT NOT NULL CHECK (target_type IN ('message', 'item_version')),
    target_id TEXT NOT NULL,
    model_name TEXT NOT NULL,
    dim INTEGER NOT NULL,
    vector BLOB NOT NULL,
    PRIMARY KEY (target_type, target_id)
);

CREATE INDEX IF NOT EXISTS idx_embeddings_type ON embeddings (target_type, model_name);

CREATE TABLE IF NOT EXISTS turn_evidences (
    assistant_message_id TEXT NOT NULL REFERENCES messages(id) ON DELETE CASCADE,
    ref_type TEXT NOT NULL CHECK (ref_type IN ('item', 'message', 'ambiguity')),
    ref_id TEXT NOT NULL,
    PRIMARY KEY (assistant_message_id, ref_type, ref_id)
);

CREATE INDEX IF NOT EXISTS idx_turn_evidences_ref ON turn_evidences (ref_type, ref_id);

CREATE TABLE IF NOT EXISTS consolidations (
    id TEXT PRIMARY KEY,
    namespace TEXT NOT NULL DEFAULT 'default',
    topic TEXT NOT NULL,
    summary TEXT NOT NULL,
    source_item_ids_json TEXT NOT NULL DEFAULT '[]',
    source_message_ids_json TEXT NOT NULL DEFAULT '[]',
    confidence REAL NOT NULL DEFAULT 0.85,
    updated_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_consolidations_updated ON consolidations (updated_at DESC);

CREATE TABLE IF NOT EXISTS consolidation_embeddings (
    consolidation_id TEXT PRIMARY KEY REFERENCES consolidations(id) ON DELETE CASCADE,
    namespace TEXT NOT NULL DEFAULT 'default',
    model_name TEXT NOT NULL,
    dim INTEGER NOT NULL,
    vector BLOB NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_cons_emb_ns ON consolidation_embeddings (namespace, model_name);

CREATE VIRTUAL TABLE IF NOT EXISTS consolidations_fts USING fts5(
    consolidation_id UNINDEXED,
    tokens
);

CREATE TABLE IF NOT EXISTS session_summaries (
    id TEXT PRIMARY KEY,
    namespace TEXT NOT NULL DEFAULT 'default',
    session_id TEXT NOT NULL,
    session_date TEXT NOT NULL,
    summary TEXT NOT NULL,
    facts_json TEXT NOT NULL DEFAULT '[]',
    entities_json TEXT NOT NULL DEFAULT '[]',
    source_message_ids_json TEXT NOT NULL DEFAULT '[]',
    updated_at TEXT NOT NULL,
    UNIQUE (namespace, session_id)
);

CREATE INDEX IF NOT EXISTS idx_session_summaries_ns
    ON session_summaries (namespace, session_date ASC);

CREATE TABLE IF NOT EXISTS session_summary_embeddings (
    summary_id TEXT PRIMARY KEY REFERENCES session_summaries(id) ON DELETE CASCADE,
    namespace TEXT NOT NULL DEFAULT 'default',
    model_name TEXT NOT NULL,
    dim INTEGER NOT NULL,
    vector BLOB NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_ss_emb_ns ON session_summary_embeddings (namespace, model_name);

CREATE VIRTUAL TABLE IF NOT EXISTS session_summaries_fts USING fts5(
    summary_id UNINDEXED,
    tokens
);

CREATE TABLE IF NOT EXISTS core_memory_blocks (
    namespace TEXT NOT NULL DEFAULT 'default',
    block_key TEXT NOT NULL,
    content TEXT NOT NULL,
    source_item_ids_json TEXT NOT NULL DEFAULT '[]',
    updated_at TEXT NOT NULL,
    PRIMARY KEY (namespace, block_key)
);

CREATE VIRTUAL TABLE IF NOT EXISTS messages_fts USING fts5(
    message_id UNINDEXED,
    tokens
);

CREATE VIRTUAL TABLE IF NOT EXISTS items_fts USING fts5(
    item_id UNINDEXED,
    version_id UNINDEXED,
    tokens
);
"""


def normalize_namespace(namespace: str | None) -> str:
    cleaned = (namespace or "default").strip().lower()
    return cleaned[:64] if cleaned else "default"


def _now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _fts_tokens(text: str) -> str:
    tokens = tokenize(text)
    return " ".join(tokens) if tokens else text.strip()


def _timed_fts_tokens(text: str, recorded_at: str) -> str:
    return " ".join([_fts_tokens(text), *extract_anchored_search_tokens(text, recorded_at)]).strip()


def _fts_expand_token(token: str) -> list[str]:
    """Lightweight English morphological query expansion (singular/base forms) for FTS5."""
    out = [token]
    if token.isascii() and token.isalpha() and len(token) >= 4:
        if token.endswith("ies") and len(token) >= 5:
            out.append(token[:-3] + "y")
        elif token.endswith("es") and len(token) >= 5 and token[-3] in {"s", "x", "z", "h"}:
            out.append(token[:-2])
        elif token.endswith("s") and not token.endswith("ss"):
            out.append(token[:-1])
        if token.endswith("ed") and len(token) >= 5:
            out.append(token[:-2])
            out.append(token[:-1])
        elif token.endswith("ing") and len(token) >= 6:
            out.append(token[:-3])
    return out


def _fts_query(text: str) -> str:
    raw_tokens = list(dict.fromkeys(tokenize(text)))[:18]
    expanded: list[str] = []
    seen: set[str] = set()
    for tok in raw_tokens:
        for cand in _fts_expand_token(tok):
            clean = cand.replace('"', "").strip()
            if clean and clean not in seen:
                seen.add(clean)
                expanded.append(f'"{clean}"')
    if not expanded:
        return ""
    return " OR ".join(expanded[:28])


class StaleExtraction(ValueError):
    """The source was received before the item's current version."""


class LocalMemoryStore:
    """Authoritative SQLite store for raw turns, evolving items, edges, vectors, and FTS5."""

    def __init__(self, db_path: Path | str, *, embedder: Embedder | None = None):
        self.db_path = Path(db_path)
        if str(self.db_path) != ":memory:":
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.embedder = embedder
        self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(SCHEMA_SQL)
        # Use the legacy insertion order once to migrate databases without a durable sequence.
        with self._conn:
            self._conn.execute(
                """
                INSERT INTO message_ingest_order (message_id)
                SELECT m.id FROM messages AS m
                WHERE NOT EXISTS (
                    SELECT 1 FROM message_ingest_order AS seq WHERE seq.message_id = m.id
                )
                ORDER BY m.rowid
                """
            )
        # Backward-compatible optional columns for provenance mapping, epistemic layering & multi-namespace.
        cols = {r["name"] for r in self._conn.execute("PRAGMA table_info(messages)").fetchall()}
        if "extract_attempts" not in cols:
            self._conn.execute("ALTER TABLE messages ADD COLUMN extract_attempts INTEGER NOT NULL DEFAULT 0")
        if "extraction_failure" not in cols:
            self._conn.execute("ALTER TABLE messages ADD COLUMN extraction_failure TEXT")
        if "external_id" not in cols:
            self._conn.execute("ALTER TABLE messages ADD COLUMN external_id TEXT")
            self._conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_messages_external_id ON messages (external_id)"
            )
        if "namespace" not in cols:
            self._conn.execute(
                "ALTER TABLE messages ADD COLUMN namespace TEXT NOT NULL DEFAULT 'default'"
            )
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_messages_ns_sess ON messages (namespace, session_id, created_at)"
        )

        item_cols = {r["name"] for r in self._conn.execute("PRAGMA table_info(items)").fetchall()}
        if "epistemic_type" not in item_cols:
            self._conn.execute(
                "ALTER TABLE items ADD COLUMN epistemic_type TEXT NOT NULL DEFAULT 'confirmed_fact'"
            )
        if "confidence" not in item_cols:
            self._conn.execute(
                "ALTER TABLE items ADD COLUMN confidence REAL NOT NULL DEFAULT 1.0"
            )
        if "salience" not in item_cols:
            self._conn.execute(
                "ALTER TABLE items ADD COLUMN salience REAL NOT NULL DEFAULT 1.0"
            )
        if "access_count" not in item_cols:
            self._conn.execute(
                "ALTER TABLE items ADD COLUMN access_count INTEGER NOT NULL DEFAULT 0"
            )
        if "namespace" not in item_cols:
            self._conn.execute(
                "ALTER TABLE items ADD COLUMN namespace TEXT NOT NULL DEFAULT 'default'"
            )
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_items_ns_status ON items (namespace, status, updated_at DESC)"
        )

        amb_cols = {r["name"] for r in self._conn.execute("PRAGMA table_info(ambiguities)").fetchall()}
        if "namespace" not in amb_cols:
            self._conn.execute(
                "ALTER TABLE ambiguities ADD COLUMN namespace TEXT NOT NULL DEFAULT 'default'"
            )

        cons_cols = {r["name"] for r in self._conn.execute("PRAGMA table_info(consolidations)").fetchall()}
        if "namespace" not in cons_cols:
            self._conn.execute(
                "ALTER TABLE consolidations ADD COLUMN namespace TEXT NOT NULL DEFAULT 'default'"
            )
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_consolidations_ns ON consolidations (namespace, updated_at DESC)"
        )
        self._conn.commit()
        self._upgrade_text_indexes()
        # In-memory acceleration caches (invalidated atomically on write/merge/forget).
        self._vec_cache: dict[tuple[str, str, str], tuple[list[str], object, dict[str, str]]] = {}
        self._graph_cache: dict[str, object] = {}
        self._query_vec_cache: dict[tuple[str, str], list[float]] = {}

    def _upgrade_text_indexes(self) -> None:
        """Rebuild pretokenized indexes atomically, and refresh legacy local hashes on use."""
        specs = (
            ("message", "messages_fts", "message_id", """
                SELECT id, content AS text, created_at AS anchor FROM messages
                WHERE role = 'user' AND deleted = 0
            """),
            ("item_version", "items_fts", "version_id", """
                SELECT v.id, v.item_id, i.title || ' ' || v.summary || ' '
                       || COALESCE(v.effective_time_text, '') || ' ' || COALESCE(v.condition_text, '') AS text,
                       i.entities_json, NULL AS anchor
                FROM item_versions AS v JOIN items AS i ON i.id = v.item_id
            """),
            ("session_summary", "session_summaries_fts", "summary_id", """
                SELECT id, summary AS text, facts_json, entities_json,
                       session_date AS anchor FROM session_summaries
            """),
            ("consolidation", "consolidations_fts", "consolidation_id", """
                SELECT id, topic || ' ' || summary AS text, NULL AS anchor FROM consolidations
            """),
        )
        hash_embedder = self.embedder if isinstance(self.embedder, HashingEmbedder) else None
        with self._conn:
            fts_changed = self._conn.execute(
                """
                INSERT INTO memory_metadata (key, value) VALUES ('tokenizer_version', ?)
                ON CONFLICT(key) DO UPDATE SET value = excluded.value
                WHERE value != excluded.value
                """,
                (str(TOKENIZER_VERSION),),
            ).rowcount == 1
            hashes_changed = hash_embedder is not None and self._conn.execute(
                "INSERT OR IGNORE INTO memory_metadata (key, value) VALUES (?, 'ready')",
                (hash_embedder.name,),
            ).rowcount == 1
            if not fts_changed and not hashes_changed:
                return
            for kind, fts_table, id_column, query in specs:
                if kind in {"message", "item_version"}:
                    vector_table = "embeddings"
                    vector_id = "target_id"
                    legacy_filter = "target_type = ? AND model_name = ?"
                    legacy_params = (kind, hash_embedder.legacy_name) if hash_embedder else ()
                else:
                    vector_table = kind + "_embeddings"
                    vector_id = id_column
                    legacy_filter = "model_name = ?"
                    legacy_params = (hash_embedder.legacy_name,) if hash_embedder else ()
                legacy_ids = set()
                if hashes_changed:
                    legacy_ids = {
                        row[0] for row in self._conn.execute(
                            f"SELECT {vector_id} FROM {vector_table} WHERE {legacy_filter}",
                            legacy_params,
                        )
                    }
                if not fts_changed and not legacy_ids:
                    continue
                if fts_changed:
                    indexed_messages = {
                        row[0] for row in self._conn.execute("SELECT message_id FROM messages_fts")
                    } if kind == "message" else set()
                    self._conn.execute(f"DELETE FROM {fts_table}")
                for row in self._conn.execute(query):
                    identity = row["id"]
                    fields = row.keys()
                    parts = [row["text"]]
                    if "facts_json" in fields:
                        parts.extend(json.loads(row["facts_json"]))
                    if "entities_json" in fields:
                        parts.extend(json.loads(row["entities_json"]))
                    content = " ".join(parts).strip()
                    if fts_changed and (kind != "message" or identity in indexed_messages):
                        tokens = _timed_fts_tokens(content, row["anchor"]) if row["anchor"] else _fts_tokens(content)
                        if kind == "item_version":
                            self._conn.execute(
                                "INSERT INTO items_fts (item_id, version_id, tokens) VALUES (?, ?, ?)",
                                (row["item_id"], identity, tokens),
                            )
                        else:
                            self._conn.execute(
                                f"INSERT INTO {fts_table} ({id_column}, tokens) VALUES (?, ?)",
                                (identity, tokens),
                            )
                    if identity in legacy_ids and hash_embedder is not None:
                        [vector] = hash_embedder.embed_passages([content])
                        self._conn.execute(
                            f"UPDATE {vector_table} SET model_name = ?, dim = ?, vector = ?"
                            f" WHERE {vector_id} = ? AND {legacy_filter}",
                            (hash_embedder.name, hash_embedder.dimension, pack_vector(vector), identity, *legacy_params),
                        )
                if legacy_ids:
                    self._conn.execute(f"DELETE FROM {vector_table} WHERE {legacy_filter}", legacy_params)

    def _invalidate_cache(self, namespace: str | None = None) -> None:
        if namespace is None:
            self._vec_cache.clear()
            self._graph_cache.clear()
            return
        ns = normalize_namespace(namespace)
        for key in [k for k in self._vec_cache if k[0] == ns]:
            self._vec_cache.pop(key, None)
        self._graph_cache.pop(ns, None)

    def _embed_query_cached(self, query: str) -> list[float]:
        if self.embedder is None or not query.strip():
            return []
        key = (self.embedder.name, query.strip())
        cached = self._query_vec_cache.get(key)
        if cached is not None:
            return cached
        vec = self.embedder.embed_query(query)
        if len(self._query_vec_cache) >= 256:
            self._query_vec_cache.clear()
        self._query_vec_cache[key] = vec
        return vec

    def close(self) -> None:
        self._conn.close()

    # ------------------------------------------------------------------
    # Messages (Immutable Event Log)
    # ------------------------------------------------------------------

    def add_message(
        self,
        session_id: str,
        role: str,
        content: str,
        *,
        namespace: str = "default",
        timezone_name: str = "UTC",
        created_at: str | None = None,
        reply_to_message_id: str | None = None,
        index_for_retrieval: bool = True,
        external_id: str | None = None,
        extract_state: bool = True,
    ) -> MessageRecord:
        ns = normalize_namespace(namespace)
        msg_id = str(uuid4())
        ts = created_at or _now_iso()
        extracted = int(role != "user" or not extract_state)
        vector_blob: bytes | None = None
        if index_for_retrieval and role == "user" and self.embedder is not None:
            vecs = self.embedder.embed_passages([content])
            if vecs and vecs[0]:
                vector_blob = pack_vector(vecs[0])
        with self._conn:
            self._conn.execute(
                """
                INSERT INTO messages (
                    id, session_id, namespace, role, content, created_at, timezone,
                    extracted, deleted, reply_to_message_id, external_id
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 0, ?, ?)
                """,
                (
                    msg_id,
                    session_id,
                    ns,
                    role,
                    content,
                    ts,
                    timezone_name,
                    extracted,
                    reply_to_message_id,
                    external_id,
                ),
            )
            if index_for_retrieval and role == "user":
                anchor_toks = extract_anchored_search_tokens(content, ts)
                fts_text = _fts_tokens(content)
                if anchor_toks:
                    fts_text = f"{fts_text} {' '.join(anchor_toks)}".strip()
                self._conn.execute(
                    "INSERT INTO messages_fts (message_id, tokens) VALUES (?, ?)",
                    (msg_id, fts_text),
                )
                if vector_blob is not None and self.embedder is not None:
                    self._conn.execute(
                        """
                        INSERT OR REPLACE INTO embeddings (
                            target_type, target_id, model_name, dim, vector
                        ) VALUES ('message', ?, ?, ?, ?)
                        """,
                        (msg_id, self.embedder.name, self.embedder.dimension, vector_blob),
                    )
        if index_for_retrieval and role == "user":
            self._invalidate_cache(ns)
        return MessageRecord(
            id=msg_id,
            session_id=session_id,
            namespace=ns,
            role=role,  # type: ignore[arg-type]
            content=content,
            created_at=ts,
            timezone=timezone_name,
            extracted=bool(extracted),
            deleted=False,
            extraction_status="done" if extracted else "pending",
        )

    def get_message(self, message_id: str) -> MessageRecord | None:
        row = self._conn.execute(
            "SELECT * FROM messages WHERE id = ? AND deleted = 0",
            (message_id,),
        ).fetchone()
        return self._row_to_message(row) if row else None

    def find_recent_duplicate_message(
        self,
        session_id: str,
        content: str,
        *,
        namespace: str = "default",
        within_seconds: int = 20,
    ) -> MessageRecord | None:
        """Detect identical user message retries within a short time window (AIRI retry idempotency)."""
        ns = normalize_namespace(namespace)
        row = self._conn.execute(
            """
            SELECT m.* FROM messages AS m
            JOIN message_ingest_order AS seq ON seq.message_id = m.id
            WHERE m.namespace = ? AND m.session_id = ? AND m.role = 'user' AND m.deleted = 0
            ORDER BY seq.sequence DESC
            LIMIT 1
            """,
            (ns, session_id),
        ).fetchone()
        if row is None or str(row["content"]).strip() != content.strip():
            return None
        try:
            prev_dt = datetime.fromisoformat(str(row["created_at"]).replace("Z", "+00:00"))
            if prev_dt.tzinfo is None:
                prev_dt = prev_dt.replace(tzinfo=UTC)
            now_dt = datetime.now(UTC)
            if abs((now_dt - prev_dt).total_seconds()) <= within_seconds:
                return self._row_to_message(row)
        except ValueError:
            return None
        return None

    def get_external_ids(self, message_ids: list[str] | tuple[str, ...]) -> dict[str, str]:
        if not message_ids:
            return {}
        ph = ",".join("?" for _ in message_ids)
        rows = self._conn.execute(
            f"SELECT id, external_id FROM messages WHERE id IN ({ph}) AND external_id IS NOT NULL",
            list(message_ids),
        ).fetchall()
        return {r["id"]: r["external_id"] for r in rows}

    def get_message_ids_by_external(
        self, external_ids: list[str] | tuple[str, ...], *, namespace: str | None = None
    ) -> list[str]:
        if not external_ids:
            return []
        ph = ",".join("?" for _ in external_ids)
        ns = normalize_namespace(namespace) if namespace is not None else None
        rows = self._conn.execute(
            f"""
            SELECT id, external_id FROM messages
            WHERE external_id IN ({ph}) AND deleted = 0 AND (? IS NULL OR namespace = ?)
            """,
            [*external_ids, ns, ns],
        ).fetchall()
        by_ext = {r["external_id"]: r["id"] for r in rows}
        return [by_ext[ext] for ext in external_ids if ext in by_ext]

    def get_session_messages(
        self, session_id: str, *, namespace: str = "default", limit: int = 12
    ) -> list[MessageRecord]:
        ns = normalize_namespace(namespace)
        rows = self._conn.execute(
            """
            SELECT * FROM (
                SELECT rowid AS _rid, * FROM messages
                WHERE namespace = ? AND session_id = ? AND deleted = 0
                ORDER BY created_at DESC, rowid DESC
                LIMIT ?
            ) ORDER BY created_at ASC, _rid ASC
            """,
            (ns, session_id, limit),
        ).fetchall()
        return [self._row_to_message(r) for r in rows]

    def list_sessions(self, *, namespace: str = "default", limit: int = 40) -> list[dict]:
        ns = normalize_namespace(namespace)
        rows = self._conn.execute(
            """
            SELECT
                session_id,
                COUNT(*) AS message_count,
                MIN(created_at) AS started_at,
                MAX(created_at) AS updated_at,
                (
                    SELECT content FROM messages AS m2
                    WHERE m2.namespace = m1.namespace AND m2.session_id = m1.session_id
                      AND m2.role = 'user' AND m2.deleted = 0
                    ORDER BY m2.created_at ASC, m2.rowid ASC
                    LIMIT 1
                ) AS first_user_message
            FROM messages AS m1
            WHERE namespace = ? AND deleted = 0
            GROUP BY session_id
            ORDER BY updated_at DESC
            LIMIT ?
            """,
            (ns, limit),
        ).fetchall()
        return [
            {
                "session_id": r["session_id"],
                "namespace": ns,
                "message_count": int(r["message_count"]),
                "started_at": r["started_at"],
                "updated_at": r["updated_at"],
                "title": ((r["first_user_message"] or "New session").strip().splitlines()[0])[:28],
            }
            for r in rows
        ]

    def list_namespaces(self) -> list[str]:
        rows = self._conn.execute(
            """
            SELECT DISTINCT namespace FROM messages WHERE deleted = 0
            UNION
            SELECT DISTINCT namespace FROM items
            UNION
            SELECT DISTINCT namespace FROM consolidations
            UNION
            SELECT DISTINCT namespace FROM core_memory_blocks
            """
        ).fetchall()
        found = sorted({str(r["namespace"]) for r in rows if r["namespace"]})
        if "default" not in found:
            found.insert(0, "default")
        return found

    # ------------------------------------------------------------------
    # Session summaries and their search indexes
    # ------------------------------------------------------------------

    def upsert_session_summary(
        self,
        session_id: str,
        *,
        namespace: str = "default",
        session_date: str,
        summary: str,
        facts: list[str] | tuple[str, ...] = (),
        entities: list[str] | tuple[str, ...] = (),
        source_message_ids: list[str] | tuple[str, ...] = (),
        updated_at: str | None = None,
    ) -> SessionSummaryRecord:
        ns = normalize_namespace(namespace)
        ts = updated_at or _now_iso()
        clean_facts = [str(f).strip() for f in facts if str(f).strip()][:16]
        clean_ents = list(dict.fromkeys(str(e).strip() for e in entities if str(e).strip()))[:20]
        clean_msgs = list(dict.fromkeys(str(m).strip() for m in source_message_ids if str(m).strip()))
        combined_text = " ".join([summary.strip(), *clean_facts, *clean_ents]).strip()

        vector_blob: bytes | None = None
        if self.embedder is not None and combined_text:
            vecs = self.embedder.embed_passages([combined_text[:2000]])
            if vecs and vecs[0]:
                vector_blob = pack_vector(vecs[0])

        with self._conn:
            existing = self._conn.execute(
                "SELECT id FROM session_summaries WHERE namespace = ? AND session_id = ?",
                (ns, session_id),
            ).fetchone()
            sid_pk = existing["id"] if existing else str(uuid4())
            if existing:
                self._conn.execute(
                    "DELETE FROM session_summaries_fts WHERE summary_id = ?", (sid_pk,)
                )
                self._conn.execute(
                    """
                    UPDATE session_summaries
                    SET session_date = ?, summary = ?, facts_json = ?, entities_json = ?,
                        source_message_ids_json = ?, updated_at = ?
                    WHERE id = ?
                    """,
                    (
                        session_date,
                        summary.strip(),
                        json.dumps(clean_facts, ensure_ascii=False),
                        json.dumps(clean_ents, ensure_ascii=False),
                        json.dumps(clean_msgs, ensure_ascii=False),
                        ts,
                        sid_pk,
                    ),
                )
            else:
                self._conn.execute(
                    """
                    INSERT INTO session_summaries (
                        id, namespace, session_id, session_date, summary,
                        facts_json, entities_json, source_message_ids_json, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        sid_pk,
                        ns,
                        session_id,
                        session_date,
                        summary.strip(),
                        json.dumps(clean_facts, ensure_ascii=False),
                        json.dumps(clean_ents, ensure_ascii=False),
                        json.dumps(clean_msgs, ensure_ascii=False),
                        ts,
                    ),
                )
            anchor_toks = extract_anchored_search_tokens(combined_text, session_date)
            fts_text = _fts_tokens(combined_text)
            if anchor_toks:
                fts_text = f"{fts_text} {' '.join(anchor_toks)}".strip()
            self._conn.execute(
                "INSERT INTO session_summaries_fts (summary_id, tokens) VALUES (?, ?)",
                (sid_pk, fts_text),
            )
            if vector_blob is not None and self.embedder is not None:
                self._conn.execute(
                    """
                    INSERT OR REPLACE INTO session_summary_embeddings (
                        summary_id, namespace, model_name, dim, vector
                    ) VALUES (?, ?, ?, ?, ?)
                    """,
                    (sid_pk, ns, self.embedder.name, self.embedder.dimension, vector_blob),
                )
        self._invalidate_cache(ns)

        return SessionSummaryRecord(
            id=sid_pk,
            namespace=ns,
            session_id=session_id,
            session_date=session_date,
            summary=summary.strip(),
            facts=tuple(clean_facts),
            entities=tuple(clean_ents),
            source_message_ids=tuple(clean_msgs),
            updated_at=ts,
        )

    def list_session_summaries(self, *, namespace: str = "default") -> list[SessionSummaryRecord]:
        ns = normalize_namespace(namespace)
        rows = self._conn.execute(
            "SELECT * FROM session_summaries WHERE namespace = ? ORDER BY session_date ASC, session_id ASC",
            (ns,),
        ).fetchall()
        return [
            SessionSummaryRecord(
                id=r["id"],
                namespace=r["namespace"],
                session_id=r["session_id"],
                session_date=r["session_date"],
                summary=r["summary"],
                facts=tuple(json.loads(r["facts_json"])),
                entities=tuple(json.loads(r["entities_json"])),
                source_message_ids=tuple(json.loads(r["source_message_ids_json"])),
                updated_at=r["updated_at"],
            )
            for r in rows
        ]

    def search_session_summaries(
        self, query: str, *, namespace: str = "default", limit: int = 8
    ) -> list[SessionSummaryRecord]:
        """Hybrid ONNX vector + FTS5 BM25 RRF search over coreference-resolved session summaries."""
        ns = normalize_namespace(namespace)
        all_ss = self.list_session_summaries(namespace=ns)
        if not all_ss:
            return []
        by_id = {s.id: s for s in all_ss}
        scores: dict[str, float] = {}

        if self.embedder is not None and query.strip():
            q_vec = self._embed_query_cached(query)
            emb_rows = self._conn.execute(
                """
                SELECT summary_id, vector
                FROM session_summary_embeddings
                WHERE namespace = ? AND model_name = ?
                """,
                (ns, self.embedder.name),
            ).fetchall()
            if emb_rows:
                hits = batch_cosine_top_k(
                    q_vec,
                    [(r["summary_id"], r["vector"]) for r in emb_rows],
                    top_k=limit * 2,
                    min_score=0.20,
                )
                for rank, (sid_pk, sim) in enumerate(hits, start=1):
                    if sid_pk in by_id:
                        scores[sid_pk] = scores.get(sid_pk, 0.0) + (1.0 / (25.0 + rank)) + 0.04 * sim

        fts_q = _fts_query(query)
        if fts_q:
            fts_rows = self._conn.execute(
                """
                SELECT f.summary_id, bm25(session_summaries_fts) AS rank
                FROM session_summaries_fts AS f
                JOIN session_summaries AS s ON s.id = f.summary_id
                WHERE session_summaries_fts MATCH ? AND s.namespace = ?
                ORDER BY rank ASC
                LIMIT ?
                """,
                (fts_q, ns, limit * 2),
            ).fetchall()
            for rank, fr in enumerate(fts_rows, start=1):
                sid_pk = fr["summary_id"]
                if sid_pk in by_id:
                    scores[sid_pk] = scores.get(sid_pk, 0.0) + (1.0 / (25.0 + rank))

        if not scores:
            return []
        ranked = sorted(scores.keys(), key=lambda k: (-scores[k], by_id[k].session_date))
        return [by_id[k] for k in ranked[:limit]]

    def get_session_summary(
        self, session_id: str, *, namespace: str = "default"
    ) -> SessionSummaryRecord | None:
        ns = normalize_namespace(namespace)
        row = self._conn.execute(
            "SELECT * FROM session_summaries WHERE namespace = ? AND session_id = ?",
            (ns, session_id),
        ).fetchone()
        if row is None:
            return None
        return SessionSummaryRecord(
            id=row["id"],
            namespace=row["namespace"],
            session_id=row["session_id"],
            session_date=row["session_date"],
            summary=row["summary"],
            facts=tuple(json.loads(row["facts_json"])),
            entities=tuple(json.loads(row["entities_json"])),
            source_message_ids=tuple(json.loads(row["source_message_ids_json"])),
            updated_at=row["updated_at"],
        )

    def count_session_user_messages(
        self, session_id: str, *, namespace: str = "default"
    ) -> int:
        ns = normalize_namespace(namespace)
        row = self._conn.execute(
            "SELECT COUNT(*) AS n FROM messages WHERE namespace = ? AND session_id = ? AND role = 'user' AND deleted = 0",
            (ns, session_id),
        ).fetchone()
        return int(row["n"]) if row else 0

    def match_known_entities(
        self, query: str, *, namespace: str = "default"
    ) -> list[str]:
        """Match known stored entities (people, pets, relationships, objects) present in query (<3ms)."""
        ns = normalize_namespace(namespace)
        q_low = query.lower()
        q_toks = set(tokenize(query))
        if not q_low.strip():
            return []

        raw_pool: set[str] = set()
        for r in self._conn.execute(
            "SELECT entities_json FROM items WHERE namespace = ?", (ns,)
        ).fetchall():
            for e in json.loads(r["entities_json"] or "[]"):
                s = str(e).strip().lower()
                if len(s) >= 2:
                    raw_pool.add(s)
        for r in self._conn.execute(
            "SELECT entities_json FROM session_summaries WHERE namespace = ?", (ns,)
        ).fetchall():
            for e in json.loads(r["entities_json"] or "[]"):
                s = str(e).strip().lower()
                if len(s) >= 2:
                    raw_pool.add(s)

        # Common English stopwords / ultra-generic words that should not act as entity anchors.
        generic_stop = {
            "user", "assistant", "today", "yesterday", "tomorrow", "week", "month", "year",
            "time", "work", "home", "life", "city", "new", "plan", "plans", "day", "night",
            "weekend", "things", "something", "anything", "people", "friend", "friends",
        }
        matched: list[str] = []
        for ent in sorted(raw_pool, key=lambda s: (-len(s), s)):
            if ent in generic_stop:
                continue
            # Match whole-token/substring for multi-word or CJK, or morphological form for ASCII.
            forms = _fts_expand_token(ent) if (" " not in ent and ent.isascii()) else [ent]
            hit = False
            for f in forms:
                if len(f) < 2:
                    continue
                if f.isascii() and " " not in f:
                    if f in q_toks or f"{f}s" in q_toks:
                        hit = True
                        break
                elif f in q_low:
                    hit = True
                    break
            if hit and ent not in matched:
                matched.append(ent)
            if len(matched) >= 8:
                break
        return matched

    @staticmethod
    def _entity_present_in_text(entity: str, text_low: str) -> bool:
        if entity in text_low:
            return True
        if entity == "girlfriend" and (" gf " in f" {text_low} " or "my gf" in text_low):
            return True
        if entity == "boyfriend" and (" bf " in f" {text_low} " or "my bf" in text_low):
            return True
        if entity.isascii() and " " not in entity:
            for form in _fts_expand_token(entity):
                if len(form) >= 3 and form in text_low:
                    return True
        return False

    def search_session_summaries_by_entities(
        self, query: str, *, namespace: str = "default", limit: int = 12
    ) -> list[SessionSummaryRecord]:
        """Entity-scoped session summary recall: surfaces all cross-session episodes for matched people/entities."""
        ns = normalize_namespace(namespace)
        matched_ents = self.match_known_entities(query, namespace=ns)
        if not matched_ents:
            return []
        all_ss = self.list_session_summaries(namespace=ns)
        if not all_ss:
            return []

        sem_scores: dict[str, float] = {}
        if self.embedder is not None and query.strip():
            q_vec = self._embed_query_cached(query)
            emb_rows = self._conn.execute(
                "SELECT summary_id, vector FROM session_summary_embeddings WHERE namespace = ? AND model_name = ?",
                (ns, self.embedder.name),
            ).fetchall()
            for sid_pk, sc in batch_cosine_top_k(
                q_vec,
                [(r["summary_id"], r["vector"]) for r in emb_rows],
                top_k=len(all_ss),
                min_score=0.0,
            ):
                sem_scores[sid_pk] = sc

        q_toks = set(tokenize(query))
        blobs = [
            f"{ss.summary} {' '.join(ss.facts)} {' '.join(ss.entities)}".lower()
            for ss in all_ss
        ]
        total_n = len(all_ss)
        ent_idf: dict[str, float] = {}
        for e in matched_ents:
            df = sum(1 for b in blobs if self._entity_present_in_text(e, b))
            ent_idf[e] = math.log(1.0 + total_n / (1.0 + df))

        scored: list[tuple[float, SessionSummaryRecord]] = []
        for ss, blob in zip(all_ss, blobs, strict=True):
            matched_here = [e for e in matched_ents if self._entity_present_in_text(e, blob)]
            if not matched_here:
                continue
            idf_score = sum(ent_idf[e] for e in matched_here)
            # Bonus when multiple distinct queried entities co-occur in the same session!
            co_bonus = 0.65 * (len(matched_here) - 1) if len(matched_here) >= 2 else 0.0
            lex = len(q_toks & set(tokenize(blob))) / max(1, len(q_toks))
            sem = sem_scores.get(ss.id, 0.0)
            score = idf_score + co_bonus + sem * 1.8 + lex * 0.5
            scored.append((score, ss))
        scored.sort(key=lambda pair: (-pair[0], pair[1].session_date))
        return [ss for _, ss in scored[:limit]]

    def search_consolidations_by_entities(
        self, query: str, *, namespace: str = "default", limit: int = 6
    ) -> list[ConsolidationRecord]:
        """IDF-weighted entity-scoped dossier recall: prioritizes specific entity + semantic matches."""
        ns = normalize_namespace(namespace)
        matched_ents = self.match_known_entities(query, namespace=ns)
        if not matched_ents:
            return []
        all_cons = self.list_consolidations(namespace=ns)
        if not all_cons:
            return []

        sem_scores: dict[str, float] = {}
        if self.embedder is not None and query.strip():
            q_vec = self._embed_query_cached(query)
            emb_rows = self._conn.execute(
                "SELECT consolidation_id, vector FROM consolidation_embeddings WHERE namespace = ? AND model_name = ?",
                (ns, self.embedder.name),
            ).fetchall()
            for cid, sc in batch_cosine_top_k(
                q_vec,
                [(r["consolidation_id"], r["vector"]) for r in emb_rows],
                top_k=len(all_cons),
                min_score=0.0,
            ):
                sem_scores[cid] = sc

        blobs = [f"{c.topic} {c.summary}".lower() for c in all_cons]
        total_n = len(all_cons)
        ent_idf = {
            e: math.log(1.0 + total_n / (1.0 + sum(1 for b in blobs if self._entity_present_in_text(e, b))))
            for e in matched_ents
        }
        q_toks = set(tokenize(query))
        scored: list[tuple[float, ConsolidationRecord]] = []
        for cons, blob in zip(all_cons, blobs, strict=True):
            matched_here = [e for e in matched_ents if self._entity_present_in_text(e, blob)]
            if not matched_here:
                continue
            idf_score = sum(ent_idf[e] for e in matched_here)
            lex = len(q_toks & set(tokenize(blob))) / max(1, len(q_toks))
            sem = sem_scores.get(cons.id, 0.0)
            scored.append((idf_score + sem * 2.0 + lex * 0.6 + 0.1 * cons.confidence, cons))
        scored.sort(key=lambda pair: (-pair[0], -pair[1].confidence))
        return [c for _, c in scored[:limit]]

    # ------------------------------------------------------------------
    # Core memory blocks
    # ------------------------------------------------------------------

    def get_core_memory_blocks(self, *, namespace: str = "default") -> list[CoreMemoryBlock]:
        ns = normalize_namespace(namespace)
        rows = self._conn.execute(
            "SELECT * FROM core_memory_blocks WHERE namespace = ? ORDER BY block_key ASC",
            (ns,),
        ).fetchall()
        return [
            CoreMemoryBlock(
                namespace=r["namespace"],
                block_key=r["block_key"],
                content=r["content"],
                source_item_ids=tuple(json.loads(r["source_item_ids_json"])),
                updated_at=r["updated_at"],
            )
            for r in rows
        ]

    def upsert_core_memory_block(
        self,
        block_key: str,
        content: str,
        *,
        namespace: str = "default",
        source_item_ids: list[str] | tuple[str, ...] = (),
        updated_at: str | None = None,
    ) -> CoreMemoryBlock:
        ns = normalize_namespace(namespace)
        key = block_key.strip().lower()[:40]
        clean_content = content.strip()[:800]
        src_items = list(dict.fromkeys(str(x).strip() for x in source_item_ids if str(x).strip()))
        ts = updated_at or _now_iso()
        with self._conn:
            self._conn.execute(
                """
                INSERT OR REPLACE INTO core_memory_blocks (
                    namespace, block_key, content, source_item_ids_json, updated_at
                ) VALUES (?, ?, ?, ?, ?)
                """,
                (ns, key, clean_content, json.dumps(src_items, ensure_ascii=False), ts),
            )
        return CoreMemoryBlock(
            namespace=ns,
            block_key=key,
            content=clean_content,
            source_item_ids=tuple(src_items),
            updated_at=ts,
        )

    def delete_core_memory_block(self, block_key: str, *, namespace: str = "default") -> bool:
        ns = normalize_namespace(namespace)
        key = block_key.strip().lower()[:40]
        with self._conn:
            cur = self._conn.execute(
                "DELETE FROM core_memory_blocks WHERE namespace = ? AND block_key = ?",
                (ns, key),
            )
        return cur.rowcount > 0

    def get_neighbor_user_messages(
        self, message_id: str, *, radius: int = 1
    ) -> list[MessageRecord]:
        row = self._conn.execute(
            "SELECT rowid, namespace, session_id FROM messages WHERE id = ? AND deleted = 0",
            (message_id,),
        ).fetchone()
        if row is None:
            return []
        ns = row["namespace"]
        session_id = row["session_id"]
        rows = self._conn.execute(
            """
            SELECT rowid, * FROM messages
            WHERE namespace = ? AND session_id = ? AND role = 'user' AND deleted = 0
            ORDER BY created_at ASC, rowid ASC
            """,
            (ns, session_id),
        ).fetchall()
        idx = next((i for i, r in enumerate(rows) if r["id"] == message_id), None)
        if idx is None:
            return []
        lo = max(0, idx - radius)
        hi = min(len(rows), idx + radius + 1)
        return [self._row_to_message(rows[i]) for i in range(lo, hi) if i != idx]

    def get_pending_messages(
        self, *, namespace: str = "default", exclude_id: str | None = None, limit: int = -1
    ) -> list[MessageRecord]:
        rows = self._conn.execute(
            """
            SELECT m.* FROM messages AS m
            JOIN message_ingest_order AS seq ON seq.message_id = m.id
            WHERE m.namespace = ? AND m.role = 'user' AND m.extracted = 0 AND m.deleted = 0
              AND m.extraction_failure IS NULL AND (? IS NULL OR m.id != ?)
            ORDER BY seq.sequence
            LIMIT ?
            """,
            (normalize_namespace(namespace), exclude_id, exclude_id, limit),
        ).fetchall()
        return [self._row_to_message(row) for row in rows]

    def record_extraction_failure(self, message_id: str, *, stale: bool = False) -> bool:
        """Count structural failures and quarantine at the limit or on a stale write."""
        with self._conn:
            self._conn.execute(
                """
                UPDATE messages
                SET extract_attempts = extract_attempts + CASE WHEN ? THEN 0 ELSE 1 END,
                    extraction_failure = CASE
                        WHEN ? THEN 'stale_source'
                        WHEN extract_attempts >= 2 THEN 'invalid_output'
                        ELSE NULL
                    END
                WHERE id = ? AND extracted = 0 AND deleted = 0 AND extraction_failure IS NULL
                """,
                (stale, stale, message_id),
            )
        message = self.get_message(message_id)
        return message is None or message.extraction_status == "quarantined"

    def list_quarantined_messages(self, *, namespace: str = "default") -> list[MessageRecord]:
        rows = self._conn.execute(
            """
            SELECT m.* FROM messages AS m
            JOIN message_ingest_order AS seq ON seq.message_id = m.id
            WHERE m.namespace = ? AND m.deleted = 0 AND m.extraction_failure IS NOT NULL
            ORDER BY seq.sequence
            """,
            (normalize_namespace(namespace),),
        ).fetchall()
        return [self._row_to_message(row) for row in rows]

    def requeue_extraction(self, message_id: str, *, namespace: str) -> bool:
        with self._conn:
            result = self._conn.execute(
                """
                UPDATE messages SET extract_attempts = 0, extraction_failure = NULL
                WHERE id = ? AND namespace = ? AND deleted = 0 AND extraction_failure IS NOT NULL
                """,
                (message_id, normalize_namespace(namespace)),
            )
        return result.rowcount == 1

    def get_messages_before(self, message: MessageRecord, *, limit: int) -> list[MessageRecord]:
        rows = self._conn.execute(
            """
            SELECT m.* FROM messages AS m
            JOIN message_ingest_order AS seq ON seq.message_id = m.id
            WHERE m.namespace = ? AND m.session_id = ? AND m.deleted = 0
              AND seq.sequence < (
                  SELECT sequence FROM message_ingest_order WHERE message_id = ?
              )
            ORDER BY seq.sequence DESC
            LIMIT ?
            """,
            (message.namespace, message.session_id, message.id, limit),
        ).fetchall()
        return [self._row_to_message(row) for row in reversed(rows)]

    def mark_message_extracted(self, message_id: str) -> None:
        with self._conn:
            self._conn.execute("UPDATE messages SET extracted = 1 WHERE id = ?", (message_id,))

    def record_turn_evidences(
        self,
        assistant_message_id: str,
        *,
        item_ids: tuple[str, ...],
        source_message_ids: tuple[str, ...],
        ambiguity_ids: tuple[str, ...],
    ) -> None:
        now = _now_iso()
        with self._conn:
            for iid in item_ids:
                self._conn.execute(
                    """
                    INSERT OR IGNORE INTO turn_evidences (assistant_message_id, ref_type, ref_id)
                    VALUES (?, 'item', ?)
                    """,
                    (assistant_message_id, iid),
                )
                self._conn.execute(
                    """
                    UPDATE items
                    SET last_mentioned_at = ?,
                        access_count = COALESCE(access_count, 0) + 1,
                        salience = MIN(1.6, COALESCE(salience, 1.0) + 0.04)
                    WHERE id = ?
                    """,
                    (now, iid),
                )
            for mid in source_message_ids:
                self._conn.execute(
                    """
                    INSERT OR IGNORE INTO turn_evidences (assistant_message_id, ref_type, ref_id)
                    VALUES (?, 'message', ?)
                    """,
                    (assistant_message_id, mid),
                )
            for aid in ambiguity_ids:
                self._conn.execute(
                    """
                    INSERT OR IGNORE INTO turn_evidences (assistant_message_id, ref_type, ref_id)
                    VALUES (?, 'ambiguity', ?)
                    """,
                    (assistant_message_id, aid),
                )

    # ------------------------------------------------------------------
    # Items, Versions, Edges, Ambiguities (Atomic Write Transaction)
    # ------------------------------------------------------------------

    def commit_extraction(
        self,
        source_message_id: str,
        operations: list[dict],
        *,
        created_at: str | None = None,
    ) -> list[str]:
        """Apply one source's entire extraction once and mark it complete in the same transaction."""
        ts = created_at or _now_iso()
        # Pre-compute embeddings outside the SQLite write lock so the transaction stays sub-ms.
        texts_to_embed: list[str] = []
        embed_indices: list[int] = []
        for idx, op in enumerate(operations):
            if op["kind"] in {"create_item", "add_version"}:
                text = " ".join(
                    part
                    for part in (
                        op.get("title") or "",
                        op.get("summary") or "",
                        op.get("effective_time_text") or "",
                        op.get("condition_text") or "",
                        " ".join(op.get("entities") or ()),
                    )
                    if part
                ).strip()
                texts_to_embed.append(text)
                embed_indices.append(idx)

        vectors: dict[int, bytes] = {}
        if texts_to_embed and self.embedder is not None:
            raw_vecs = self.embedder.embed_passages(texts_to_embed)
            for op_idx, vec in zip(embed_indices, raw_vecs, strict=True):
                if vec:
                    vectors[op_idx] = pack_vector(vec)

        touched_item_ids: list[str] = []
        with self._conn:
            # Claim the source with a write first so simultaneous commits cannot both apply it.
            claimed = self._conn.execute(
                """
                UPDATE messages SET extracted = 1
                WHERE id = ? AND role = 'user' AND deleted = 0 AND extracted = 0
                  AND extraction_failure IS NULL
                """,
                (source_message_id,),
            )
            if claimed.rowcount == 0:
                return []
            msg_row = self._conn.execute(
                """
                SELECT m.namespace, seq.sequence FROM messages AS m
                JOIN message_ingest_order AS seq ON seq.message_id = m.id WHERE m.id = ?
                """,
                (source_message_id,),
            ).fetchone()
            ns = normalize_namespace(msg_row["namespace"])

            for idx, op in enumerate(operations):
                op_kind = op["kind"]
                if op_kind == "create_item":
                    item_id = str(uuid4())
                    ver_id = str(uuid4())
                    title = op["title"]
                    kind: ItemKind = op["item_kind"]
                    status: ItemStatus = op["status"]
                    entities = list(dict.fromkeys(op.get("entities") or []))
                    self._conn.execute(
                        """
                        INSERT INTO items (
                            id, namespace, title, kind, status, current_version_id,
                            entities_json, epistemic_type, confidence, created_at, updated_at
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            item_id,
                            ns,
                            title,
                            kind,
                            status,
                            ver_id,
                            json.dumps(entities, ensure_ascii=False),
                            op.get("epistemic_type") or ("shared_commitment" if kind == "plan" else "confirmed_fact"),
                            float(op.get("confidence", 1.0)),
                            ts,
                            ts,
                        ),
                    )
                    v_from = op.get("valid_from") or ts
                    self._conn.execute(
                        """
                        INSERT INTO item_versions (
                            id, item_id, version_no, summary, status,
                            effective_time_text, valid_from, valid_to, condition_text,
                            change_type, supersedes_version_id,
                            source_message_id, source_quote, created_at
                        ) VALUES (?, ?, 1, ?, ?, ?, ?, ?, ?, ?, NULL, ?, ?, ?)
                        """,
                        (
                            ver_id,
                            item_id,
                            op["summary"],
                            status,
                            op.get("effective_time_text"),
                            v_from,
                            op.get("valid_to"),
                            op.get("condition_text"),
                            op["change_type"],
                            source_message_id,
                            op["source_quote"],
                            ts,
                        ),
                    )
                    self._insert_edge_tx(
                        "item_version", ver_id, "message", source_message_id,
                        "DERIVED_FROM", source_message_id, ts,
                    )
                    for rel_id in op.get("related_item_ids") or ():
                        if self._item_exists_tx(rel_id) and rel_id != item_id:
                            self._insert_edge_tx(
                                "item", item_id, "item", rel_id,
                                "RELATED_ENTITY", source_message_id, ts,
                            )
                    self._index_version_tx(
                        item_id,
                        ver_id,
                        title,
                        op["summary"],
                        op.get("effective_time_text"),
                        op.get("condition_text"),
                        entities,
                        vectors.get(idx),
                    )
                    self._auto_link_zettelkasten_tx(
                        item_id=item_id,
                        entities=entities,
                        vec_blob=vectors.get(idx),
                        namespace=ns,
                        source_message_id=source_message_id,
                        ts=ts,
                    )
                    touched_item_ids.append(item_id)

                elif op_kind == "add_version":
                    item_id = op["item_id"]
                    item_row = self._conn.execute(
                        """
                        SELECT i.*, seq.sequence AS source_sequence FROM items AS i
                        LEFT JOIN item_versions AS v ON v.id = i.current_version_id
                        LEFT JOIN message_ingest_order AS seq ON seq.message_id = v.source_message_id
                        WHERE i.id = ? AND i.namespace = ?
                        """,
                        (item_id, ns),
                    ).fetchone()
                    if item_row is None:
                        continue
                    if item_row["source_sequence"] is not None and item_row["source_sequence"] > msg_row["sequence"]:
                        raise StaleExtraction("A newer source already updated this item")
                    prev_ver_id = item_row["current_version_id"]
                    max_no = self._conn.execute(
                        "SELECT COALESCE(MAX(version_no), 0) AS n FROM item_versions WHERE item_id = ?",
                        (item_id,),
                    ).fetchone()["n"]
                    ver_id = str(uuid4())
                    new_status: ItemStatus = op["status"]
                    existing_entities = json.loads(item_row["entities_json"])
                    merged_entities = list(
                        dict.fromkeys([*existing_entities, *(op.get("entities") or [])])
                    )
                    v_from = op.get("valid_from") or ts
                    if prev_ver_id:
                        # Close the open validity interval when a new version supersedes it.
                        self._conn.execute(
                            "UPDATE item_versions SET valid_to = COALESCE(valid_to, ?) WHERE id = ?",
                            (v_from, prev_ver_id),
                        )
                    self._conn.execute(
                        """
                        INSERT INTO item_versions (
                            id, item_id, version_no, summary, status,
                            effective_time_text, valid_from, valid_to, condition_text,
                            change_type, supersedes_version_id,
                            source_message_id, source_quote, created_at
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            ver_id,
                            item_id,
                            int(max_no) + 1,
                            op["summary"],
                            new_status,
                            op.get("effective_time_text"),
                            v_from,
                            op.get("valid_to"),
                            op.get("condition_text"),
                            op["change_type"],
                            prev_ver_id,
                            source_message_id,
                            op["source_quote"],
                            ts,
                        ),
                    )
                    self._conn.execute(
                        """
                        UPDATE items
                        SET status = ?, current_version_id = ?, entities_json = ?, updated_at = ?
                        WHERE id = ?
                        """,
                        (
                            new_status,
                            ver_id,
                            json.dumps(merged_entities, ensure_ascii=False),
                            ts,
                            item_id,
                        ),
                    )
                    self._insert_edge_tx(
                        "item_version", ver_id, "message", source_message_id,
                        "DERIVED_FROM", source_message_id, ts,
                    )
                    if prev_ver_id:
                        rel: EdgeRelation = (
                            "RETRACTS"
                            if op["change_type"] in {"retract", "cancel"}
                            else "SUPERSEDES"
                        )
                        self._insert_edge_tx(
                            "item_version", ver_id, "item_version", prev_ver_id,
                            rel, source_message_id, ts,
                        )
                    for rel_id in op.get("related_item_ids") or ():
                        if self._item_exists_tx(rel_id) and rel_id != item_id:
                            self._insert_edge_tx(
                                "item", item_id, "item", rel_id,
                                "RELATED_ENTITY", source_message_id, ts,
                            )
                    # Resolve open ambiguities that targeted this item.
                    self._resolve_ambiguities_for_item_tx(item_id)
                    self._index_version_tx(
                        item_id,
                        ver_id,
                        item_row["title"],
                        op["summary"],
                        op.get("effective_time_text"),
                        op.get("condition_text"),
                        merged_entities,
                        vectors.get(idx),
                    )
                    self._auto_link_zettelkasten_tx(
                        item_id=item_id,
                        entities=merged_entities,
                        vec_blob=vectors.get(idx),
                        namespace=ns,
                        source_message_id=source_message_id,
                        ts=ts,
                    )
                    touched_item_ids.append(item_id)

                elif op_kind == "add_ambiguity":
                    amb_id = str(uuid4())
                    self._conn.execute(
                        """
                        INSERT INTO ambiguities (
                            id, namespace, source_message_id, source_quote,
                            candidate_item_ids_json, note, resolved, created_at
                        ) VALUES (?, ?, ?, ?, ?, ?, 0, ?)
                        """,
                        (
                            amb_id,
                            ns,
                            source_message_id,
                            op["source_quote"],
                            json.dumps(list(op.get("candidate_item_ids") or []), ensure_ascii=False),
                            op["note"],
                            ts,
                        ),
                    )

        if touched_item_ids:
            self._invalidate_cache(ns)
        return touched_item_ids

    def _auto_link_zettelkasten_tx(
        self,
        *,
        item_id: str,
        entities: list[str],
        vec_blob: bytes | None,
        namespace: str,
        source_message_id: str,
        ts: str,
        max_links: int = 3,
    ) -> int:
        """Link items with related entities during extraction."""
        from hmph.local_memory.graph import normalize_entity

        norm_ents = {normalize_entity(str(e)) for e in entities}
        norm_ents.discard("")
        other_rows = self._conn.execute(
            """
            SELECT i.id, i.entities_json, i.current_version_id
            FROM items AS i
            WHERE i.namespace = ? AND i.id != ? AND i.status IN ('active', 'paused')
            """,
            (namespace, item_id),
        ).fetchall()
        if not other_rows:
            return 0

        existing_neighbors: set[str] = set()
        for er in self._conn.execute(
            """
            SELECT src_id, dst_id FROM memory_edges
            WHERE src_type = 'item' AND dst_type = 'item' AND (src_id = ? OR dst_id = ?)
            """,
            (item_id, item_id),
        ).fetchall():
            existing_neighbors.add(er["dst_id"] if er["src_id"] == item_id else er["src_id"])

        cand_scores: dict[str, float] = {}
        for row in other_rows:
            oid = row["id"]
            if oid in existing_neighbors:
                continue
            o_ents = {normalize_entity(str(e)) for e in json.loads(row["entities_json"] or "[]")}
            o_ents.discard("")
            shared = norm_ents & o_ents
            if shared:
                cand_scores[oid] = cand_scores.get(oid, 0.0) + 0.55 * len(shared)

        if vec_blob is not None and self.embedder is not None:
            src_vec = unpack_vector(vec_blob)
            ver_ids = [r["current_version_id"] for r in other_rows if r["current_version_id"]]
            if ver_ids:
                ph = ",".join("?" for _ in ver_ids)
                vrows = self._conn.execute(
                    f"""
                    SELECT target_id, vector FROM embeddings
                    WHERE target_type = 'item_version' AND model_name = ? AND target_id IN ({ph})
                    """,
                    [self.embedder.name, *ver_ids],
                ).fetchall()
                ver_to_oid = {
                    r["current_version_id"]: r["id"]
                    for r in other_rows
                    if r["current_version_id"]
                }
                for vr in vrows:
                    oid = ver_to_oid.get(vr["target_id"])
                    if not oid or oid in existing_neighbors:
                        continue
                    sim = cosine_similarity(src_vec, unpack_vector(vr["vector"]))
                    if 0.60 <= sim < 0.84:
                        cand_scores[oid] = cand_scores.get(oid, 0.0) + sim

        if not cand_scores:
            return 0
        top_oids = sorted(cand_scores.keys(), key=lambda k: (-cand_scores[k], k))[:max_links]
        for oid in top_oids:
            self._insert_edge_tx(
                "item", item_id, "item", oid, "RELATED_ENTITY", source_message_id, ts
            )
        return len(top_oids)

    def _item_exists_tx(self, item_id: str) -> bool:
        row = self._conn.execute("SELECT 1 FROM items WHERE id = ?", (item_id,)).fetchone()
        return row is not None

    def _insert_edge_tx(
        self,
        src_type: str,
        src_id: str,
        dst_type: str,
        dst_id: str,
        relation: EdgeRelation,
        source_message_id: str | None,
        created_at: str,
    ) -> None:
        self._conn.execute(
            """
            INSERT INTO memory_edges (
                id, src_type, src_id, dst_type, dst_id, relation, source_message_id, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (str(uuid4()), src_type, src_id, dst_type, dst_id, relation, source_message_id, created_at),
        )

    def _index_version_tx(
        self,
        item_id: str,
        version_id: str,
        title: str,
        summary: str,
        effective_time_text: str | None,
        condition_text: str | None,
        entities: list[str],
        vector_blob: bytes | None,
    ) -> None:
        combined = " ".join(
            part
            for part in (title, summary, effective_time_text or "", condition_text or "", " ".join(entities))
            if part
        )
        self._conn.execute(
            "INSERT INTO items_fts (item_id, version_id, tokens) VALUES (?, ?, ?)",
            (item_id, version_id, _fts_tokens(combined)),
        )
        if vector_blob is not None and self.embedder is not None:
            self._conn.execute(
                """
                INSERT OR REPLACE INTO embeddings (
                    target_type, target_id, model_name, dim, vector
                ) VALUES ('item_version', ?, ?, ?, ?)
                """,
                (version_id, self.embedder.name, self.embedder.dimension, vector_blob),
            )

    def _resolve_ambiguities_for_item_tx(self, item_id: str) -> None:
        rows = self._conn.execute(
            "SELECT id, candidate_item_ids_json FROM ambiguities WHERE resolved = 0"
        ).fetchall()
        for row in rows:
            cands = json.loads(row["candidate_item_ids_json"])
            if item_id in cands:
                self._conn.execute("UPDATE ambiguities SET resolved = 1 WHERE id = ?", (row["id"],))

    # ------------------------------------------------------------------
    # Querying & Multi-Channel Hybrid Retrieval
    # ------------------------------------------------------------------

    def list_items(
        self, *, namespace: str = "default", status: ItemStatus | None = None
    ) -> list[ItemRecord]:
        ns = normalize_namespace(namespace)
        if status:
            rows = self._conn.execute(
                "SELECT * FROM items WHERE namespace = ? AND status = ? ORDER BY updated_at DESC",
                (ns, status),
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT * FROM items WHERE namespace = ? ORDER BY updated_at DESC",
                (ns,),
            ).fetchall()
        return [self._hydrate_item(row) for row in rows]

    def get_item(self, item_id: str) -> ItemRecord | None:
        row = self._conn.execute("SELECT * FROM items WHERE id = ?", (item_id,)).fetchone()
        return self._hydrate_item(row) if row else None

    def list_open_ambiguities(
        self, *, namespace: str = "default", limit: int = 5
    ) -> list[AmbiguityRecord]:
        ns = normalize_namespace(namespace)
        rows = self._conn.execute(
            """
            SELECT * FROM ambiguities
            WHERE namespace = ? AND resolved = 0
            ORDER BY created_at DESC
            LIMIT ?
            """,
            (ns, limit),
        ).fetchall()
        return [self._row_to_ambiguity(r) for r in rows]

    def list_edges_for_items(self, item_ids: list[str]) -> list[EdgeRecord]:
        if not item_ids:
            return []
        placeholders = ",".join("?" for _ in item_ids)
        rows = self._conn.execute(
            f"""
            SELECT * FROM memory_edges
            WHERE (src_type = 'item' AND src_id IN ({placeholders}))
               OR (dst_type = 'item' AND dst_id IN ({placeholders}))
            """,
            [*item_ids, *item_ids],
        ).fetchall()
        return [
            EdgeRecord(
                id=r["id"],
                src_type=r["src_type"],
                src_id=r["src_id"],
                dst_type=r["dst_type"],
                dst_id=r["dst_id"],
                relation=r["relation"],
                source_message_id=r["source_message_id"],
                created_at=r["created_at"],
            )
            for r in rows
        ]

    def search_items_fts(
        self, query: str, *, namespace: str = "default", limit: int = 30
    ) -> list[str]:
        fts_q = _fts_query(query)
        if not fts_q:
            return []
        ns = normalize_namespace(namespace)
        rows = self._conn.execute(
            """
            SELECT f.item_id, bm25(items_fts) AS rank
            FROM items_fts AS f
            JOIN items AS i ON i.id = f.item_id
            WHERE items_fts MATCH ? AND i.namespace = ?
            ORDER BY rank ASC
            LIMIT ?
            """,
            (fts_q, ns, limit),
        ).fetchall()
        return list(dict.fromkeys(r["item_id"] for r in rows))

    def search_messages_fts(
        self, query: str, *, namespace: str = "default", limit: int = 30
    ) -> list[str]:
        fts_q = _fts_query(query)
        if not fts_q:
            return []
        ns = normalize_namespace(namespace)
        rows = self._conn.execute(
            """
            SELECT f.message_id, bm25(messages_fts) AS rank
            FROM messages_fts AS f
            JOIN messages AS m ON m.id = f.message_id
            WHERE messages_fts MATCH ? AND m.deleted = 0 AND m.namespace = ?
            ORDER BY rank ASC
            LIMIT ?
            """,
            (fts_q, ns, limit),
        ).fetchall()
        return list(dict.fromkeys(r["message_id"] for r in rows))

    def _get_cached_vector_index(
        self, namespace: str, target_type: str
    ) -> tuple[list[str], object, dict[str, str]]:
        """Build or return cached L2-normalized Float32 matrix for (namespace, target_type, model)."""
        assert self.embedder is not None
        ns = normalize_namespace(namespace)
        cache_key = (ns, target_type, self.embedder.name)
        cached = self._vec_cache.get(cache_key)
        if cached is not None:
            return cached
        if target_type == "item_version":
            rows = self._conn.execute(
                """
                SELECT e.target_id, e.vector, v.item_id
                FROM embeddings AS e
                JOIN item_versions AS v ON v.id = e.target_id
                JOIN items AS i ON i.id = v.item_id
                WHERE e.target_type = 'item_version' AND e.model_name = ? AND i.namespace = ?
                """,
                (self.embedder.name, ns),
            ).fetchall()
            aux = {r["target_id"]: r["item_id"] for r in rows}
            ids, mat = build_normalized_matrix(
                [(r["target_id"], r["vector"]) for r in rows], self.embedder.dimension
            )
        else:
            rows = self._conn.execute(
                """
                SELECT e.target_id, e.vector
                FROM embeddings AS e
                JOIN messages AS m ON m.id = e.target_id
                WHERE e.target_type = 'message' AND e.model_name = ? AND m.deleted = 0 AND m.namespace = ?
                """,
                (self.embedder.name, ns),
            ).fetchall()
            aux = {}
            ids, mat = build_normalized_matrix(
                [(r["target_id"], r["vector"]) for r in rows], self.embedder.dimension
            )
        entry = (ids, mat, aux)
        self._vec_cache[cache_key] = entry
        return entry

    def search_items_semantic_scored(
        self,
        query: str,
        *,
        namespace: str = "default",
        limit: int = 30,
        min_score: float = 0.18,
    ) -> list[tuple[str, float]]:
        if self.embedder is None or not query.strip():
            return []
        q_vec = self._embed_query_cached(query)
        ids, mat, ver_to_item = self._get_cached_vector_index(namespace, "item_version")
        if not ids:
            return []
        ranked = matrix_cosine_top_k(q_vec, ids, mat, top_k=limit * 2, min_score=min_score)
        best_by_item: dict[str, float] = {}
        for vid, sc in ranked:
            iid = ver_to_item.get(vid)
            if iid and sc > best_by_item.get(iid, -1.0):
                best_by_item[iid] = sc
        ordered = sorted(best_by_item.items(), key=lambda pair: (-pair[1], pair[0]))
        return ordered[:limit]

    def search_items_semantic(
        self,
        query: str,
        *,
        namespace: str = "default",
        limit: int = 30,
        min_score: float = 0.18,
    ) -> list[str]:
        return [
            iid
            for iid, _ in self.search_items_semantic_scored(
                query, namespace=namespace, limit=limit, min_score=min_score
            )
        ]

    def search_messages_semantic_scored(
        self,
        query: str,
        *,
        namespace: str = "default",
        limit: int = 30,
        min_score: float = 0.18,
    ) -> list[tuple[str, float]]:
        if self.embedder is None or not query.strip():
            return []
        q_vec = self._embed_query_cached(query)
        ids, mat, _ = self._get_cached_vector_index(namespace, "message")
        if not ids:
            return []
        return matrix_cosine_top_k(q_vec, ids, mat, top_k=limit, min_score=min_score)

    def search_messages_semantic(
        self, query: str, *, namespace: str = "default", limit: int = 30
    ) -> list[str]:
        return [
            mid
            for mid, _ in self.search_messages_semantic_scored(
                query, namespace=namespace, limit=limit
            )
        ]

    def count_user_messages(self, *, namespace: str = "default") -> int:
        ns = normalize_namespace(namespace)
        row = self._conn.execute(
            "SELECT COUNT(*) AS n FROM messages WHERE namespace = ? AND role = 'user' AND deleted = 0",
            (ns,),
        ).fetchone()
        return int(row["n"]) if row else 0

    def rebuild_fts_anchors(self) -> int:
        """Rebuild messages_fts with relative-date/year anchored tokens (zero LLM cost)."""
        rows = self._conn.execute(
            "SELECT id, content, created_at FROM messages WHERE role = 'user' AND deleted = 0"
        ).fetchall()
        with self._conn:
            self._conn.execute("DELETE FROM messages_fts")
            for r in rows:
                anchor_toks = extract_anchored_search_tokens(r["content"], r["created_at"])
                fts_text = _fts_tokens(r["content"])
                if anchor_toks:
                    fts_text = f"{fts_text} {' '.join(anchor_toks)}".strip()
                self._conn.execute(
                    "INSERT INTO messages_fts (message_id, tokens) VALUES (?, ?)",
                    (r["id"], fts_text),
                )
        return len(rows)

    def expand_one_hop_items(self, item_ids: list[str], *, limit: int = 8) -> list[str]:
        if not item_ids:
            return []
        seed = set(item_ids)
        neighbors: list[str] = []
        for edge in self.list_edges_for_items(item_ids):
            if edge.src_type == "item" and edge.dst_type == "item":
                other = edge.dst_id if edge.src_id in seed else edge.src_id
                if other not in seed and other not in neighbors:
                    neighbors.append(other)
                    if len(neighbors) >= limit:
                        break
        return neighbors

    def find_similar_item(
        self,
        text: str,
        *,
        namespace: str = "default",
        kind: str | None = None,
        threshold: float = 0.84,
    ) -> ItemRecord | None:
        """Find an existing item in this namespace whose current version is semantically near-duplicate."""
        if self.embedder is None or not text.strip():
            return None
        ns = normalize_namespace(namespace)
        q_vec = self.embedder.embed_query(text)
        if kind:
            rows = self._conn.execute(
                """
                SELECT e.target_id, e.vector, i.id AS item_id
                FROM embeddings AS e
                JOIN items AS i ON i.current_version_id = e.target_id
                WHERE e.target_type = 'item_version' AND e.model_name = ? AND i.namespace = ? AND i.kind = ?
                """,
                (self.embedder.name, ns, kind),
            ).fetchall()
        else:
            rows = self._conn.execute(
                """
                SELECT e.target_id, e.vector, i.id AS item_id
                FROM embeddings AS e
                JOIN items AS i ON i.current_version_id = e.target_id
                WHERE e.target_type = 'item_version' AND e.model_name = ? AND i.namespace = ?
                """,
                (self.embedder.name, ns),
            ).fetchall()
        if not rows:
            return None
        ver_to_item = {r["target_id"]: r["item_id"] for r in rows}
        ranked = batch_cosine_top_k(
            q_vec,
            [(r["target_id"], r["vector"]) for r in rows],
            top_k=1,
            min_score=threshold,
        )
        if not ranked:
            return None
        return self.get_item(ver_to_item[ranked[0][0]])

    def _get_cached_graph(self, namespace: str):
        from hmph.local_memory.graph import MemoryBipartiteGraph

        ns = normalize_namespace(namespace)
        cached = self._graph_cache.get(ns)
        if cached is not None:
            return cached
        graph = MemoryBipartiteGraph.from_sqlite(self._conn, namespace=ns)
        self._graph_cache[ns] = graph
        return graph

    def run_heterogeneous_ppr(
        self,
        *,
        query: str,
        seed_item_ids: list[str],
        seed_message_ids: list[str],
        seed_summary_ids: list[str] | tuple[str, ...] = (),
        namespace: str = "default",
        top_items: int = 12,
        top_messages: int = 16,
        top_summaries: int = 8,
    ) -> tuple[list[tuple[str, float]], list[tuple[str, float]], list[tuple[str, float]]]:
        graph = self._get_cached_graph(namespace)
        return graph.personalized_pagerank_full(
            query=query,
            seed_item_ids=seed_item_ids,
            seed_message_ids=seed_message_ids,
            seed_summary_ids=seed_summary_ids,
            top_items=top_items,
            top_messages=top_messages,
            top_summaries=top_summaries,
        )

    def run_bipartite_ppr(
        self,
        *,
        query: str,
        seed_item_ids: list[str],
        seed_message_ids: list[str],
        namespace: str = "default",
        top_items: int = 12,
        top_messages: int = 16,
    ) -> tuple[list[tuple[str, float]], list[tuple[str, float]]]:
        items, msgs, _ = self.run_heterogeneous_ppr(
            query=query,
            seed_item_ids=seed_item_ids,
            seed_message_ids=seed_message_ids,
            namespace=namespace,
            top_items=top_items,
            top_messages=top_messages,
        )
        return items, msgs

    # ------------------------------------------------------------------
    # Sleep Consolidation, Maintenance & Graph Export
    # ------------------------------------------------------------------

    def replace_consolidations(
        self,
        insights: list[dict],
        *,
        namespace: str = "default",
        updated_at: str | None = None,
    ) -> list[ConsolidationRecord]:
        """Atomically replace consolidated reflective insights for a namespace with vector + FTS5 indexing."""
        ns = normalize_namespace(namespace)
        ts = updated_at or _now_iso()
        prepared: list[dict] = []
        texts_to_embed: list[str] = []
        for ins in insights:
            topic = str(ins.get("topic") or "").strip()[:60]
            summary = str(ins.get("summary") or "").strip()[:500]
            if not topic or not summary:
                continue
            cid = str(uuid4())
            prepared.append(
                {
                    "id": cid,
                    "topic": topic,
                    "summary": summary,
                    "source_item_ids": list(dict.fromkeys(ins.get("source_item_ids") or [])),
                    "source_message_ids": list(dict.fromkeys(ins.get("source_message_ids") or [])),
                    "confidence": max(0.0, min(1.0, float(ins.get("confidence", 0.85)))),
                }
            )
            texts_to_embed.append(f"{topic} {summary}")

        vectors: list[bytes | None] = [None] * len(prepared)
        if texts_to_embed and self.embedder is not None:
            raw_vecs = self.embedder.embed_passages(texts_to_embed)
            for idx, vec in enumerate(raw_vecs):
                if vec:
                    vectors[idx] = pack_vector(vec)

        with self._conn:
            old_ids = [
                r["id"]
                for r in self._conn.execute(
                    "SELECT id FROM consolidations WHERE namespace = ?", (ns,)
                ).fetchall()
            ]
            if old_ids:
                ph = ",".join("?" for _ in old_ids)
                self._conn.execute(
                    f"DELETE FROM memory_edges WHERE src_type = 'consolidation' AND src_id IN ({ph})",
                    old_ids,
                )
                self._conn.execute(
                    f"DELETE FROM consolidations_fts WHERE consolidation_id IN ({ph})",
                    old_ids,
                )
                self._conn.execute(
                    f"DELETE FROM consolidation_embeddings WHERE consolidation_id IN ({ph})",
                    old_ids,
                )
                self._conn.execute(f"DELETE FROM consolidations WHERE id IN ({ph})", old_ids)

            for idx, entry in enumerate(prepared):
                cid = entry["id"]
                self._conn.execute(
                    """
                    INSERT INTO consolidations (
                        id, namespace, topic, summary, source_item_ids_json,
                        source_message_ids_json, confidence, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        cid,
                        ns,
                        entry["topic"],
                        entry["summary"],
                        json.dumps(entry["source_item_ids"], ensure_ascii=False),
                        json.dumps(entry["source_message_ids"], ensure_ascii=False),
                        entry["confidence"],
                        ts,
                    ),
                )
                self._conn.execute(
                    "INSERT INTO consolidations_fts (consolidation_id, tokens) VALUES (?, ?)",
                    (cid, _fts_tokens(f"{entry['topic']} {entry['summary']}")),
                )
                if vectors[idx] is not None and self.embedder is not None:
                    self._conn.execute(
                        """
                        INSERT OR REPLACE INTO consolidation_embeddings (
                            consolidation_id, namespace, model_name, dim, vector
                        ) VALUES (?, ?, ?, ?, ?)
                        """,
                        (cid, ns, self.embedder.name, self.embedder.dimension, vectors[idx]),
                    )
                for iid in entry["source_item_ids"]:
                    if self._item_exists_tx(iid):
                        self._insert_edge_tx(
                            "consolidation", cid, "item", iid, "DERIVED_FROM", None, ts
                        )
                for mid in entry["source_message_ids"]:
                    self._insert_edge_tx(
                        "consolidation", cid, "message", mid, "DERIVED_FROM", mid, ts
                    )
        self._invalidate_cache(ns)
        return self.list_consolidations(namespace=ns)

    def list_consolidations(self, *, namespace: str = "default") -> list[ConsolidationRecord]:
        ns = normalize_namespace(namespace)
        rows = self._conn.execute(
            "SELECT * FROM consolidations WHERE namespace = ? ORDER BY confidence DESC, updated_at DESC",
            (ns,),
        ).fetchall()
        results: list[ConsolidationRecord] = []
        for r in rows:
            src_items = tuple(json.loads(r["source_item_ids_json"]))
            src_msgs = tuple(json.loads(r["source_message_ids_json"]))
            quotes: list[str] = []
            if src_msgs:
                ph = ",".join("?" for _ in src_msgs)
                mrows = self._conn.execute(
                    f"SELECT content, created_at FROM messages WHERE id IN ({ph}) AND deleted = 0 ORDER BY created_at ASC",
                    list(src_msgs),
                ).fetchall()
                quotes = [f"({mr['created_at']}) {mr['content'][:180]}" for mr in mrows[:4]]
            results.append(
                ConsolidationRecord(
                    id=r["id"],
                    namespace=r["namespace"] if "namespace" in r.keys() else ns,
                    topic=r["topic"],
                    summary=r["summary"],
                    source_item_ids=src_items,
                    source_message_ids=src_msgs,
                    source_quotes=tuple(quotes),
                    confidence=float(r["confidence"]),
                    updated_at=r["updated_at"],
                )
            )
        return results

    def search_consolidations(
        self,
        query: str,
        *,
        namespace: str = "default",
        limit: int = 4,
        include_top_if_empty: bool = False,
    ) -> list[ConsolidationRecord]:
        """Hybrid vector (ONNX cosine) + FTS5 BM25 RRF search over consolidated insights."""
        ns = normalize_namespace(namespace)
        all_cons = self.list_consolidations(namespace=ns)
        if not all_cons:
            return []
        by_id = {c.id: c for c in all_cons}

        scores: dict[str, float] = {}
        # 1. Dense semantic channel (catches abstract/paraphrased queries with zero lexical overlap).
        if self.embedder is not None and query.strip():
            q_vec = self._embed_query_cached(query)
            emb_rows = self._conn.execute(
                """
                SELECT consolidation_id, vector
                FROM consolidation_embeddings
                WHERE namespace = ? AND model_name = ?
                """,
                (ns, self.embedder.name),
            ).fetchall()
            if emb_rows:
                sem_hits = batch_cosine_top_k(
                    q_vec,
                    [(r["consolidation_id"], r["vector"]) for r in emb_rows],
                    top_k=limit * 2,
                    min_score=0.22,
                )
                for rank, (cid, sim) in enumerate(sem_hits, start=1):
                    if cid in by_id:
                        scores[cid] = scores.get(cid, 0.0) + (1.0 / (25.0 + rank)) + 0.04 * sim

        # 2. FTS5 BM25 lexical channel.
        fts_q = _fts_query(query)
        if fts_q:
            fts_rows = self._conn.execute(
                """
                SELECT f.consolidation_id, bm25(consolidations_fts) AS rank
                FROM consolidations_fts AS f
                JOIN consolidations AS c ON c.id = f.consolidation_id
                WHERE consolidations_fts MATCH ? AND c.namespace = ?
                ORDER BY rank ASC
                LIMIT ?
                """,
                (fts_q, ns, limit * 2),
            ).fetchall()
            for rank, fr in enumerate(fts_rows, start=1):
                cid = fr["consolidation_id"]
                if cid in by_id:
                    scores[cid] = scores.get(cid, 0.0) + (1.0 / (25.0 + rank))

        # 3. Direct token-overlap boost (handles older DBs or HashingEmbedder).
        q_toks = set(tokenize(query))
        for cons in all_cons:
            c_toks = set(tokenize(f"{cons.topic} {cons.summary}"))
            overlap = len(q_toks & c_toks)
            if overlap > 0:
                scores[cons.id] = scores.get(cons.id, 0.0) + 0.03 * (
                    overlap / max(1, len(q_toks))
                )

        if scores:
            ranked_ids = sorted(
                scores.keys(),
                key=lambda cid: (-scores[cid], -by_id[cid].confidence),
            )
            return [by_id[cid] for cid in ranked_ids[:limit]]
        if include_top_if_empty:
            return all_cons[:limit]
        return []

    def merge_near_duplicate_items(
        self, *, namespace: str = "default", threshold: float = 0.86
    ) -> int:
        """Merge semantically near-duplicate active items of the same kind within a namespace."""
        if self.embedder is None:
            return 0
        ns = normalize_namespace(namespace)
        rows = self._conn.execute(
            """
            SELECT i.id, i.title, i.kind, i.epistemic_type, i.entities_json, i.created_at,
                   e.vector
            FROM items AS i
            JOIN embeddings AS e
              ON e.target_type = 'item_version' AND e.target_id = i.current_version_id
            WHERE i.namespace = ? AND i.status = 'active' AND e.model_name = ?
            ORDER BY i.created_at ASC
            """,
            (ns, self.embedder.name),
        ).fetchall()
        if len(rows) < 2:
            return 0

        vecs = [unpack_vector(r["vector"]) for r in rows]
        merged_away: set[str] = set()
        merged_count = 0

        with self._conn:
            for i in range(len(rows)):
                keep = rows[i]
                if keep["id"] in merged_away:
                    continue
                for j in range(i + 1, len(rows)):
                    dup = rows[j]
                    if dup["id"] in merged_away or dup["kind"] != keep["kind"]:
                        continue
                    # Never merge an inferred belief into overwriting a confirmed fact improperly;
                    # skip if epistemic trust conflicts with keep being weaker.
                    if (
                        keep["epistemic_type"] == "inferred_belief"
                        and dup["epistemic_type"] != "inferred_belief"
                    ):
                        continue
                    sim = cosine_similarity(vecs[i], vecs[j])
                    if sim < threshold:
                        continue
                    # Merge dup's versions into keep.
                    max_no = int(
                        self._conn.execute(
                            "SELECT COALESCE(MAX(version_no), 0) AS n FROM item_versions WHERE item_id = ?",
                            (keep["id"],),
                        ).fetchone()["n"]
                    )
                    dup_vers = self._conn.execute(
                        "SELECT id FROM item_versions WHERE item_id = ? ORDER BY version_no ASC",
                        (dup["id"],),
                    ).fetchall()
                    last_ver_id = None
                    for offset, vr in enumerate(dup_vers, start=1):
                        self._conn.execute(
                            "UPDATE item_versions SET item_id = ?, version_no = ? WHERE id = ?",
                            (keep["id"], max_no + offset, vr["id"]),
                        )
                        last_ver_id = vr["id"]
                    keep_ents = json.loads(keep["entities_json"])
                    dup_ents = json.loads(dup["entities_json"])
                    merged_ents = list(dict.fromkeys([*keep_ents, *dup_ents]))
                    if last_ver_id:
                        self._conn.execute(
                            "UPDATE items SET current_version_id = ?, entities_json = ?, updated_at = ? WHERE id = ?",
                            (last_ver_id, json.dumps(merged_ents, ensure_ascii=False), _now_iso(), keep["id"]),
                        )
                    self._conn.execute(
                        "UPDATE items_fts SET item_id = ? WHERE item_id = ?",
                        (keep["id"], dup["id"]),
                    )
                    self._conn.execute(
                        "UPDATE memory_edges SET src_id = ? WHERE src_type = 'item' AND src_id = ?",
                        (keep["id"], dup["id"]),
                    )
                    self._conn.execute(
                        "UPDATE memory_edges SET dst_id = ? WHERE dst_type = 'item' AND dst_id = ?",
                        (keep["id"], dup["id"]),
                    )
                    self._conn.execute(
                        "DELETE FROM memory_edges WHERE src_type = 'item' AND dst_type = 'item' AND src_id = dst_id"
                    )
                    self._conn.execute(
                        "DELETE FROM turn_evidences WHERE ref_type = 'item' AND ref_id = ?",
                        (dup["id"],),
                    )
                    self._conn.execute("DELETE FROM items WHERE id = ?", (dup["id"],))
                    merged_away.add(dup["id"])
                    merged_count += 1
        if merged_count > 0:
            self._invalidate_cache(ns)
        return merged_count

    def decay_stale_salience(
        self,
        *,
        namespace: str = "default",
        factor: float = 0.85,
        floor: float = 0.25,
        stale_days: int = 30,
    ) -> int:
        """Gently decay salience of stale non-plan episodes without hiding them from explicit recall."""
        ns = normalize_namespace(namespace)
        now_dt = datetime.now(UTC)
        rows = self._conn.execute(
            "SELECT id, updated_at, last_mentioned_at, salience FROM items WHERE namespace = ? AND kind = 'episode' AND salience > ?",
            (ns, floor),
        ).fetchall()
        decayed = 0
        with self._conn:
            for r in rows:
                ref_ts = r["last_mentioned_at"] or r["updated_at"]
                try:
                    dt = datetime.fromisoformat(str(ref_ts).replace("Z", "+00:00"))
                    if dt.tzinfo is None:
                        dt = dt.replace(tzinfo=UTC)
                    age_days = (now_dt - dt).total_seconds() / 86400.0
                except ValueError:
                    age_days = 0.0
                if age_days >= stale_days:
                    new_sal = max(floor, round(float(r["salience"]) * factor, 3))
                    self._conn.execute(
                        "UPDATE items SET salience = ? WHERE id = ?",
                        (new_sal, r["id"]),
                    )
                    decayed += 1
        return decayed

    def export_graph_data(self, *, namespace: str = "default") -> dict:
        """Export lightweight Memory Evolution Relationship Graph JSON for a namespace."""
        ns = normalize_namespace(namespace)
        items = self.list_items(namespace=ns)
        consolidations = self.list_consolidations(namespace=ns)
        nodes: list[dict] = []
        edges: list[dict] = []
        seen_edge_keys: set[tuple[str, str, str]] = set()

        def add_edge(src: str, dst: str, relation: str) -> None:
            key = (src, dst, relation)
            if src != dst and key not in seen_edge_keys:
                seen_edge_keys.add(key)
                edges.append({"source": src, "target": dst, "relation": relation})

        entity_counts: dict[str, int] = {}
        entity_label: dict[str, str] = {}
        item_entities: dict[str, list[str]] = {}

        for it in items:
            cur = it.current
            nid = f"item:{it.id}"
            nodes.append(
                {
                    "id": nid,
                    "raw_id": it.id,
                    "type": "item",
                    "label": it.title,
                    "status": it.status,
                    "kind": it.kind,
                    "summary": cur.summary if cur else it.title,
                    "version_count": len(it.versions),
                    "salience": it.salience,
                    "epistemic_type": it.epistemic_type,
                    "effective_time": cur.effective_time_text if cur else None,
                    "updated_at": it.updated_at,
                    "evolution": [
                        f"{v.change_type}: {v.summary}" for v in it.versions
                    ],
                }
            )
            ents: list[str] = []
            for ent in it.entities:
                norm = ent.strip().lower()
                if len(norm) < 2:
                    continue
                entity_counts[norm] = entity_counts.get(norm, 0) + 1
                entity_label.setdefault(norm, ent.strip())
                ents.append(norm)
            item_entities[nid] = list(dict.fromkeys(ents))

        # Keep top entities by degree so graph remains clean and readable.
        top_entities = sorted(
            entity_counts.keys(), key=lambda k: (-entity_counts[k], k)
        )[:36]
        top_entity_set = set(top_entities)
        for norm in top_entities:
            nodes.append(
                {
                    "id": f"entity:{norm}",
                    "raw_id": norm,
                    "type": "entity",
                    "label": entity_label[norm],
                    "degree": entity_counts[norm],
                }
            )
        for nid, ents in item_entities.items():
            for norm in ents:
                if norm in top_entity_set:
                    add_edge(nid, f"entity:{norm}", "MENTIONS")

        # Explicit item-to-item edges from memory_edges.
        item_id_set = {it.id for it in items}
        edge_rows = self._conn.execute(
            "SELECT src_id, dst_id, relation FROM memory_edges WHERE src_type = 'item' AND dst_type = 'item'"
        ).fetchall()
        for er in edge_rows:
            if er["src_id"] in item_id_set and er["dst_id"] in item_id_set:
                add_edge(f"item:{er['src_id']}", f"item:{er['dst_id']}", er["relation"])

        # Consolidated insights nodes & DERIVED_FROM links.
        for cons in consolidations:
            cid = f"cons:{cons.id}"
            nodes.append(
                {
                    "id": cid,
                    "raw_id": cons.id,
                    "type": "consolidation",
                    "label": cons.topic,
                    "summary": cons.summary,
                    "confidence": cons.confidence,
                    "source_quotes": list(cons.source_quotes),
                    "updated_at": cons.updated_at,
                }
            )
            for iid in cons.source_item_ids:
                if iid in item_id_set:
                    add_edge(cid, f"item:{iid}", "DERIVED_FROM")

        return {
            "namespace": ns,
            "nodes": nodes,
            "edges": edges,
            "stats": {
                "items": len(items),
                "entities": len(top_entities),
                "consolidations": len(consolidations),
                "edges": len(edges),
            },
        }

    # ------------------------------------------------------------------
    # Atomic Cascade Forgetting (DAG-based, namespace-scoped, zero resurrection)
    # ------------------------------------------------------------------

    def forget(self, target: str, *, namespace: str = "default") -> ForgetReceipt:
        """Cascade-forget an item (by ID or keyword) or message ID atomically within a namespace."""
        ns = normalize_namespace(namespace)
        target = target.strip()
        if not target:
            return ForgetReceipt(query=target, namespace=ns)

        item_ids: set[str] = set()
        message_ids: set[str] = set()
        version_ids: set[str] = set()
        ambiguity_ids: set[str] = set()

        # 1. Resolve target strictly inside this namespace.
        exact_item = self._conn.execute(
            "SELECT id FROM items WHERE id = ? AND namespace = ?", (target, ns)
        ).fetchone()
        exact_msg = self._conn.execute(
            "SELECT id FROM messages WHERE id = ? AND namespace = ? AND deleted = 0", (target, ns)
        ).fetchone()

        if exact_item:
            item_ids.add(exact_item["id"])
        elif exact_msg:
            message_ids.add(exact_msg["id"])
        else:
            like = f"%{target}%"
            rows = self._conn.execute(
                """
                SELECT DISTINCT i.id
                FROM items AS i
                LEFT JOIN item_versions AS v ON v.item_id = i.id
                WHERE i.namespace = ? AND (i.title LIKE ? OR v.summary LIKE ? OR v.source_quote LIKE ?)
                """,
                (ns, like, like, like),
            ).fetchall()
            item_ids.update(r["id"] for r in rows)
            if not item_ids:
                mrows = self._conn.execute(
                    "SELECT id FROM messages WHERE namespace = ? AND role = 'user' AND deleted = 0 AND content LIKE ?",
                    (ns, like),
                ).fetchall()
                message_ids.update(r["id"] for r in mrows)

        # 2. Expand closure: items -> versions -> source messages -> derived assistant replies & ambiguities.
        with self._conn:
            if item_ids:
                ph = ",".join("?" for _ in item_ids)
                vrows = self._conn.execute(
                    f"SELECT id, source_message_id FROM item_versions WHERE item_id IN ({ph})",
                    list(item_ids),
                ).fetchall()
                for vr in vrows:
                    version_ids.add(vr["id"])
                    message_ids.add(vr["source_message_id"])

                # Find any ambiguities referencing these items.
                arows = self._conn.execute(
                    "SELECT id, source_message_id, candidate_item_ids_json FROM ambiguities"
                ).fetchall()
                for ar in arows:
                    cands = set(json.loads(ar["candidate_item_ids_json"]))
                    if cands & item_ids:
                        ambiguity_ids.add(ar["id"])
                        message_ids.add(ar["source_message_id"])

                # Find assistant replies that used any of these items as turn evidence.
                erows = self._conn.execute(
                    f"SELECT assistant_message_id FROM turn_evidences WHERE ref_type = 'item' AND ref_id IN ({ph})",
                    list(item_ids),
                ).fetchall()
                for er in erows:
                    message_ids.add(er["assistant_message_id"])

            if message_ids:
                # Any versions derived from deleted source messages must also go.
                ph_m = ",".join("?" for _ in message_ids)
                vrows2 = self._conn.execute(
                    f"SELECT id, item_id FROM item_versions WHERE source_message_id IN ({ph_m})",
                    list(message_ids),
                ).fetchall()
                affected_items_from_msgs: set[str] = set()
                for vr in vrows2:
                    version_ids.add(vr["id"])
                    affected_items_from_msgs.add(vr["item_id"])

                # Any ambiguities from deleted messages.
                arows2 = self._conn.execute(
                    f"SELECT id FROM ambiguities WHERE source_message_id IN ({ph_m})",
                    list(message_ids),
                ).fetchall()
                ambiguity_ids.update(r["id"] for r in arows2)

                # Direct assistant replies to deleted user messages, or replies citing them.
                rrows = self._conn.execute(
                    f"SELECT id FROM messages WHERE reply_to_message_id IN ({ph_m})",
                    list(message_ids),
                ).fetchall()
                message_ids.update(r["id"] for r in rrows)

                ph_m2 = ",".join("?" for _ in message_ids)
                erows2 = self._conn.execute(
                    f"SELECT assistant_message_id FROM turn_evidences WHERE ref_type = 'message' AND ref_id IN ({ph_m2})",
                    list(message_ids),
                ).fetchall()
                message_ids.update(r["assistant_message_id"] for r in erows2)

                # Also remove user questions whose assistant reply depended ONLY on forgotten items
                # or directly asked about the forgotten items (prevent stale Q&A replay on restart).
                if item_ids:
                    ph_i = ",".join("?" for _ in item_ids)
                    linked_replies = self._conn.execute(
                        f"""
                        SELECT m.id, m.reply_to_message_id
                        FROM messages AS m
                        JOIN turn_evidences AS te ON te.assistant_message_id = m.id
                        WHERE te.ref_type = 'item' AND te.ref_id IN ({ph_i})
                        """,
                        list(item_ids),
                    ).fetchall()
                    for lr in linked_replies:
                        message_ids.add(lr["id"])
                        if lr["reply_to_message_id"]:
                            # Only delete the paired user question if it isn't a source for another surviving item.
                            other_use = self._conn.execute(
                                f"""
                                SELECT 1 FROM item_versions
                                WHERE source_message_id = ? AND item_id NOT IN ({ph_i})
                                LIMIT 1
                                """,
                                [lr["reply_to_message_id"], *item_ids],
                            ).fetchone()
                            if other_use is None:
                                message_ids.add(lr["reply_to_message_id"])

                # Delete affected versions and repair or drop partially affected items.
                if version_ids:
                    ph_v = ",".join("?" for _ in version_ids)
                    self._conn.execute(
                        f"DELETE FROM item_versions WHERE id IN ({ph_v})",
                        list(version_ids),
                    )
                    self._conn.execute(
                        f"DELETE FROM embeddings WHERE target_type = 'item_version' AND target_id IN ({ph_v})",
                        list(version_ids),
                    )
                    self._conn.execute(
                        f"DELETE FROM items_fts WHERE version_id IN ({ph_v})",
                        list(version_ids),
                    )

                for aid in affected_items_from_msgs - item_ids:
                    remaining = self._conn.execute(
                        "SELECT id, status FROM item_versions WHERE item_id = ? ORDER BY version_no DESC LIMIT 1",
                        (aid,),
                    ).fetchone()
                    if remaining is None:
                        item_ids.add(aid)
                    else:
                        self._conn.execute(
                            "UPDATE items SET current_version_id = ?, status = ?, updated_at = ? WHERE id = ?",
                            (remaining["id"], remaining["status"], _now_iso(), aid),
                        )

            deleted_edges = 0
            if item_ids:
                ph_i = ",".join("?" for _ in item_ids)
                cur = self._conn.execute(
                    f"""
                    DELETE FROM memory_edges
                    WHERE (src_type = 'item' AND src_id IN ({ph_i}))
                       OR (dst_type = 'item' AND dst_id IN ({ph_i}))
                    """,
                    [*item_ids, *item_ids],
                )
                deleted_edges += cur.rowcount
                self._conn.execute(f"DELETE FROM items_fts WHERE item_id IN ({ph_i})", list(item_ids))
                self._conn.execute(f"DELETE FROM items WHERE id IN ({ph_i})", list(item_ids))

            if version_ids:
                ph_v = ",".join("?" for _ in version_ids)
                cur = self._conn.execute(
                    f"""
                    DELETE FROM memory_edges
                    WHERE (src_type = 'item_version' AND src_id IN ({ph_v}))
                       OR (dst_type = 'item_version' AND dst_id IN ({ph_v}))
                    """,
                    [*version_ids, *version_ids],
                )
                deleted_edges += cur.rowcount

            if ambiguity_ids:
                ph_a = ",".join("?" for _ in ambiguity_ids)
                self._conn.execute(f"DELETE FROM ambiguities WHERE id IN ({ph_a})", list(ambiguity_ids))

            consolidation_ids: set[str] = set()
            if item_ids or message_ids:
                crows = self._conn.execute(
                    "SELECT id, source_item_ids_json, source_message_ids_json FROM consolidations WHERE namespace = ?",
                    (ns,),
                ).fetchall()
                for cr in crows:
                    c_items = set(json.loads(cr["source_item_ids_json"]))
                    c_msgs = set(json.loads(cr["source_message_ids_json"]))
                    if (c_items & item_ids) or (c_msgs & message_ids):
                        consolidation_ids.add(cr["id"])
                if consolidation_ids:
                    ph_c = ",".join("?" for _ in consolidation_ids)
                    cur = self._conn.execute(
                        f"DELETE FROM memory_edges WHERE src_type = 'consolidation' AND src_id IN ({ph_c})",
                        list(consolidation_ids),
                    )
                    deleted_edges += cur.rowcount
                    self._conn.execute(
                        f"DELETE FROM consolidations_fts WHERE consolidation_id IN ({ph_c})",
                        list(consolidation_ids),
                    )
                    self._conn.execute(
                        f"DELETE FROM consolidation_embeddings WHERE consolidation_id IN ({ph_c})",
                        list(consolidation_ids),
                    )
                    self._conn.execute(
                        f"DELETE FROM consolidations WHERE id IN ({ph_c})",
                        list(consolidation_ids),
                    )

            session_summary_ids: set[str] = set()
            if message_ids:
                ssrows = self._conn.execute(
                    "SELECT id, source_message_ids_json FROM session_summaries WHERE namespace = ?",
                    (ns,),
                ).fetchall()
                for sr in ssrows:
                    s_msgs = set(json.loads(sr["source_message_ids_json"]))
                    if s_msgs & message_ids:
                        session_summary_ids.add(sr["id"])
                if session_summary_ids:
                    ph_ss = ",".join("?" for _ in session_summary_ids)
                    self._conn.execute(
                        f"DELETE FROM session_summaries_fts WHERE summary_id IN ({ph_ss})",
                        list(session_summary_ids),
                    )
                    self._conn.execute(
                        f"DELETE FROM session_summary_embeddings WHERE summary_id IN ({ph_ss})",
                        list(session_summary_ids),
                    )
                    self._conn.execute(
                        f"DELETE FROM session_summaries WHERE id IN ({ph_ss})",
                        list(session_summary_ids),
                    )

            cleared_core_blocks: set[str] = set()
            if item_ids or target:
                cbrows = self._conn.execute(
                    "SELECT block_key, content, source_item_ids_json FROM core_memory_blocks WHERE namespace = ?",
                    (ns,),
                ).fetchall()
                for cbr in cbrows:
                    cb_items = set(json.loads(cbr["source_item_ids_json"]))
                    if (cb_items & item_ids) or (
                        len(target) >= 2 and target.lower() in str(cbr["content"]).lower()
                    ):
                        cleared_core_blocks.add(cbr["block_key"])
                for bk in cleared_core_blocks:
                    self._conn.execute(
                        "DELETE FROM core_memory_blocks WHERE namespace = ? AND block_key = ?",
                        (ns, bk),
                    )

            if message_ids:
                ph_m = ",".join("?" for _ in message_ids)
                cur = self._conn.execute(
                    f"""
                    DELETE FROM memory_edges
                    WHERE source_message_id IN ({ph_m})
                       OR (dst_type = 'message' AND dst_id IN ({ph_m}))
                    """,
                    [*message_ids, *message_ids],
                )
                deleted_edges += cur.rowcount
                self._conn.execute(
                    f"DELETE FROM embeddings WHERE target_type = 'message' AND target_id IN ({ph_m})",
                    list(message_ids),
                )
                self._conn.execute(
                    f"DELETE FROM messages_fts WHERE message_id IN ({ph_m})",
                    list(message_ids),
                )
                self._conn.execute(
                    f"DELETE FROM turn_evidences WHERE assistant_message_id IN ({ph_m}) OR (ref_type = 'message' AND ref_id IN ({ph_m}))",
                    [*message_ids, *message_ids],
                )
                self._conn.execute(f"DELETE FROM messages WHERE id IN ({ph_m})", list(message_ids))

        self._invalidate_cache(ns)
        return ForgetReceipt(
            query=target,
            namespace=ns,
            deleted_item_ids=tuple(sorted(item_ids)),
            deleted_version_ids=tuple(sorted(version_ids)),
            deleted_message_ids=tuple(sorted(message_ids)),
            deleted_ambiguity_ids=tuple(sorted(ambiguity_ids)),
            deleted_consolidation_ids=tuple(sorted(consolidation_ids)),
            deleted_session_summary_ids=tuple(sorted(session_summary_ids)),
            cleared_core_blocks=tuple(sorted(cleared_core_blocks)),
            deleted_edge_count=deleted_edges,
        )

    # ------------------------------------------------------------------
    # Row Hydration Helpers
    # ------------------------------------------------------------------

    def _hydrate_item(self, row: sqlite3.Row) -> ItemRecord:
        vrows = self._conn.execute(
            """
            SELECT v.*, m.content AS source_content
            FROM item_versions AS v
            LEFT JOIN messages AS m ON m.id = v.source_message_id
            WHERE v.item_id = ?
            ORDER BY v.version_no ASC
            """,
            (row["id"],),
        ).fetchall()
        versions = tuple(
            VersionRecord(
                id=vr["id"],
                item_id=vr["item_id"],
                version_no=vr["version_no"],
                summary=vr["summary"],
                status=vr["status"],
                effective_time_text=vr["effective_time_text"],
                valid_from=vr["valid_from"],
                valid_to=vr["valid_to"],
                condition_text=vr["condition_text"],
                change_type=vr["change_type"],
                supersedes_version_id=vr["supersedes_version_id"],
                source_message_id=vr["source_message_id"],
                source_quote=vr["source_quote"],
                source_content=vr["source_content"],
                created_at=vr["created_at"],
            )
            for vr in vrows
        )
        return ItemRecord(
            id=row["id"],
            namespace=row["namespace"] if "namespace" in row.keys() else "default",
            title=row["title"],
            kind=row["kind"],
            status=row["status"],
            current_version_id=row["current_version_id"],
            entities=tuple(json.loads(row["entities_json"])),
            epistemic_type=row["epistemic_type"] if "epistemic_type" in row.keys() else "confirmed_fact",
            confidence=float(row["confidence"]) if "confidence" in row.keys() else 1.0,
            salience=float(row["salience"]) if "salience" in row.keys() else 1.0,
            access_count=int(row["access_count"]) if "access_count" in row.keys() else 0,
            last_mentioned_at=row["last_mentioned_at"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            versions=versions,
        )

    # ------------------------------------------------------------------
    # Bi-Temporal Point-in-Time Snapshot, Session Idle & Export/Import
    # ------------------------------------------------------------------

    def get_latest_session_activity(
        self, base_session_id: str, *, namespace: str = "default"
    ) -> tuple[str | None, str | None, int]:
        """Return (latest_session_id, latest_created_at, episode_index) for a base session family."""
        ns = normalize_namespace(namespace)
        row = self._conn.execute(
            """
            SELECT session_id, created_at
            FROM messages
            WHERE namespace = ? AND deleted = 0
              AND (session_id = ? OR session_id LIKE ?)
            ORDER BY created_at DESC, rowid DESC
            LIMIT 1
            """,
            (ns, base_session_id, f"{base_session_id}#ep%"),
        ).fetchone()
        if row is None:
            return None, None, 1
        sid = str(row["session_id"])
        ep_no = 1
        if "#ep" in sid:
            tail = sid.rsplit("#ep", 1)[1]
            if tail.isdigit():
                ep_no = int(tail)
        return sid, str(row["created_at"]), ep_no

    @staticmethod
    def _ts_leq(left: str | None, right: str) -> bool:
        if not left:
            return True
        try:
            dt_l = datetime.fromisoformat(left.replace("Z", "+00:00"))
            dt_r = datetime.fromisoformat(right.replace("Z", "+00:00"))
            if dt_l.tzinfo is None:
                dt_l = dt_l.replace(tzinfo=UTC)
            if dt_r.tzinfo is None:
                dt_r = dt_r.replace(tzinfo=UTC)
            return dt_l <= dt_r
        except ValueError:
            dl = parse_anchor_date(left)
            dr = parse_anchor_date(right)
            if dl is not None and dr is not None:
                return dl <= dr
            return str(left) <= str(right)

    def get_items_as_of(
        self,
        as_of: str,
        *,
        namespace: str = "default",
        include_closed: bool = False,
    ) -> list[ItemRecord]:
        """Reconstruct exact bi-temporal state of all items in `namespace` as of `as_of` timestamp."""
        ns = normalize_namespace(namespace)
        all_items = self.list_items(namespace=ns)
        snapshots: list[ItemRecord] = []
        for item in all_items:
            valid_vers = [
                v
                for v in item.versions
                if self._ts_leq(v.valid_from or v.created_at, as_of)
            ]
            if not valid_vers:
                continue
            active_ver = valid_vers[-1]
            if not include_closed and active_ver.status not in {"active", "paused"}:
                continue
            snapshots.append(
                item.model_copy(
                    update={
                        "status": active_ver.status,
                        "current_version_id": active_ver.id,
                        "updated_at": active_ver.created_at,
                        "versions": tuple(valid_vers),
                    }
                )
            )
        return snapshots

    def export_namespace_data(self, *, namespace: str = "default") -> dict:
        """Export a portable, lossless JSON snapshot of a character namespace."""
        ns = normalize_namespace(namespace)
        items = self.list_items(namespace=ns)
        core_blocks = self.get_core_memory_blocks(namespace=ns)
        consolidations = self.list_consolidations(namespace=ns)
        summaries = self.list_session_summaries(namespace=ns)
        msg_rows = self._conn.execute(
            """
            SELECT * FROM messages
            WHERE namespace = ? AND deleted = 0
            ORDER BY created_at ASC, rowid ASC
            """,
            (ns,),
        ).fetchall()
        return {
            "schema_version": "0.8.0",
            "namespace": ns,
            "exported_at": _now_iso(),
            "core_blocks": [b.model_dump(mode="json") for b in core_blocks],
            "items": [it.model_dump(mode="json") for it in items],
            "consolidations": [c.model_dump(mode="json") for c in consolidations],
            "session_summaries": [s.model_dump(mode="json") for s in summaries],
            "messages": [self._row_to_message(r).model_dump(mode="json") for r in msg_rows],
        }

    def export_namespace_markdown(self, *, namespace: str = "default") -> str:
        """Render a human-readable Markdown companion memory handbook for a namespace."""
        ns = normalize_namespace(namespace)
        data = self.export_namespace_data(namespace=ns)
        lines = [
            f"# Hmph Memory Handbook — `{ns}`",
            "",
            f"- Exported at: `{data['exported_at']}`",
            f"- Items: {len(data['items'])} | Insights: {len(data['consolidations'])} | Session Summaries: {len(data['session_summaries'])}",
            "",
        ]
        if data["core_blocks"]:
            lines.append("## Core Memory")
            for blk in data["core_blocks"]:
                lines.append(f"- **{blk['block_key']}**: {blk['content']}")
            lines.append("")

        active_items = [it for it in data["items"] if it["status"] == "active"]
        paused_items = [it for it in data["items"] if it["status"] == "paused"]
        closed_items = [it for it in data["items"] if it["status"] not in {"active", "paused"}]

        if active_items:
            lines.append("## Active Commitments, Facts & Preferences")
            for it in active_items:
                cur = it["versions"][-1] if it["versions"] else {}
                eff = f" *(time: {cur.get('effective_time_text')})*" if cur.get("effective_time_text") else ""
                lines.append(f"- **[{it['kind']}] {it['title']}**: {cur.get('summary', '')}{eff}")
                if len(it["versions"]) > 1:
                    for v in it["versions"]:
                        vf = v.get("valid_from") or v.get("created_at") or ""
                        vt = v.get("valid_to") or "present"
                        lines.append(
                            f"  - {v['change_type']} → {v['status']} ({vf} to {vt}): {v['summary']}"
                        )
            lines.append("")

        if paused_items:
            lines.append("## Paused Plans (Do Not Nag)")
            for it in paused_items:
                cur = it["versions"][-1] if it["versions"] else {}
                lines.append(f"- **{it['title']}**: {cur.get('summary', '')}")
            lines.append("")

        if data["consolidations"]:
            lines.append("## Consolidated User Insights")
            for c in data["consolidations"]:
                lines.append(f"- **{c['topic']}**: {c['summary']}")
            lines.append("")

        if data["session_summaries"]:
            lines.append("## Episodic Session Summaries")
            for s in data["session_summaries"]:
                lines.append(f"- **{s['session_date']} (`{s['session_id']}`)**: {s['summary']}")
                for f in s.get("facts") or []:
                    lines.append(f"  - {f}")
            lines.append("")

        if closed_items:
            lines.append("## Historical / Superseded Items")
            for it in closed_items:
                cur = it["versions"][-1] if it["versions"] else {}
                lines.append(f"- **[{it['status']}] {it['title']}**: {cur.get('summary', '')}")
            lines.append("")

        return "\n".join(lines).rstrip() + "\n"

    def import_namespace_data(
        self, payload: dict, *, target_namespace: str | None = None
    ) -> dict[str, int]:
        """Idempotently import a JSON snapshot into `target_namespace` and rebuild search indices."""
        ns = normalize_namespace(target_namespace or payload.get("namespace") or "default")
        imported_msgs = 0
        imported_items = 0
        imported_summaries = 0
        imported_core = 0

        existing_msg_ids = {
            r["id"]
            for r in self._conn.execute(
                "SELECT id FROM messages WHERE namespace = ?", (ns,)
            ).fetchall()
        }
        msg_id_remap: dict[str, str] = {}

        for m in payload.get("messages") or ():
            orig_mid = str(m["id"])
            # If importing into a different namespace on the same DB, remap message UUIDs cleanly.
            other_row = self._conn.execute(
                "SELECT namespace FROM messages WHERE id = ?", (orig_mid,)
            ).fetchone()
            if other_row is not None and other_row["namespace"] != ns:
                new_mid = str(uuid4())
            elif orig_mid in existing_msg_ids:
                msg_id_remap[orig_mid] = orig_mid
                continue
            else:
                new_mid = orig_mid
            msg_id_remap[orig_mid] = new_mid
            role = m.get("role", "user")
            content = str(m.get("content", ""))
            created_at = str(m.get("created_at") or _now_iso())
            tz = str(m.get("timezone") or "UTC")
            failure = m.get("extraction_failure")
            if role != "user" or failure not in ("invalid_output", "stale_source"):
                failure = None
            attempts = int(m.get("extract_attempts") or 0) if failure else 0
            vec_blob = None
            if role == "user" and self.embedder is not None and content.strip():
                vecs = self.embedder.embed_passages([content])
                if vecs and vecs[0]:
                    vec_blob = pack_vector(vecs[0])
            with self._conn:
                self._conn.execute(
                    """
                    INSERT INTO messages (
                        id, session_id, namespace, role, content, created_at, timezone,
                        extracted, deleted, extract_attempts, extraction_failure
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 0, ?, ?)
                    """,
                    (
                        new_mid, str(m.get("session_id") or "imported"), ns, role, content,
                        created_at, tz, int(failure is None), attempts, failure,
                    ),
                )
                if role == "user":
                    anchor_toks = extract_anchored_search_tokens(content, created_at)
                    fts_text = _fts_tokens(content)
                    if anchor_toks:
                        fts_text = f"{fts_text} {' '.join(anchor_toks)}".strip()
                    self._conn.execute(
                        "INSERT INTO messages_fts (message_id, tokens) VALUES (?, ?)",
                        (new_mid, fts_text),
                    )
                    if vec_blob is not None and self.embedder is not None:
                        self._conn.execute(
                            "INSERT OR REPLACE INTO embeddings (target_type, target_id, model_name, dim, vector) VALUES ('message', ?, ?, ?, ?)",
                            (new_mid, self.embedder.name, self.embedder.dimension, vec_blob),
                        )
            imported_msgs += 1

        for it in payload.get("items") or ():
            orig_iid = str(it["id"])
            other_i = self._conn.execute("SELECT namespace FROM items WHERE id = ?", (orig_iid,)).fetchone()
            if other_i is not None and other_i["namespace"] == ns:
                continue
            new_iid = str(uuid4()) if other_i is not None else orig_iid
            versions = list(it.get("versions") or [])
            if not versions:
                continue
            ver_remap: dict[str, str] = {}
            for v in versions:
                orig_vid = str(v["id"])
                other_v = self._conn.execute("SELECT 1 FROM item_versions WHERE id = ?", (orig_vid,)).fetchone()
                ver_remap[orig_vid] = str(uuid4()) if other_v is not None else orig_vid
            cur_vid = ver_remap.get(str(it.get("current_version_id") or ""), ver_remap[str(versions[-1]["id"])])
            ents = list(it.get("entities") or [])
            with self._conn:
                self._conn.execute(
                    """
                    INSERT INTO items (
                        id, namespace, title, kind, status, current_version_id, entities_json,
                        epistemic_type, confidence, salience, access_count, last_mentioned_at, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        new_iid,
                        ns,
                        it["title"],
                        it["kind"],
                        it["status"],
                        cur_vid,
                        json.dumps(ents, ensure_ascii=False),
                        it.get("epistemic_type") or "confirmed_fact",
                        float(it.get("confidence", 1.0)),
                        float(it.get("salience", 1.0)),
                        int(it.get("access_count", 0)),
                        it.get("last_mentioned_at"),
                        it.get("created_at") or _now_iso(),
                        it.get("updated_at") or _now_iso(),
                    ),
                )
                for v in versions:
                    n_vid = ver_remap[str(v["id"])]
                    orig_smid = str(v["source_message_id"])
                    smid = msg_id_remap.get(orig_smid)
                    if smid is None:
                        # Synthesize fallback source message if archive omitted raw messages.
                        smid = str(uuid4())
                        q_txt = str(v.get("source_quote") or v.get("summary") or it["title"])
                        self._conn.execute(
                            "INSERT INTO messages (id, session_id, namespace, role, content, created_at, timezone, extracted, deleted) VALUES (?, 'imported', ?, 'user', ?, ?, 'UTC', 1, 0)",
                            (smid, ns, q_txt, v.get("created_at") or _now_iso()),
                        )
                        msg_id_remap[orig_smid] = smid
                    sup_vid = ver_remap.get(str(v["supersedes_version_id"])) if v.get("supersedes_version_id") else None
                    self._conn.execute(
                        """
                        INSERT INTO item_versions (
                            id, item_id, version_no, summary, status, effective_time_text,
                            valid_from, valid_to, condition_text, change_type, supersedes_version_id,
                            source_message_id, source_quote, created_at
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            n_vid,
                            new_iid,
                            int(v["version_no"]),
                            v["summary"],
                            v["status"],
                            v.get("effective_time_text"),
                            v.get("valid_from") or v.get("created_at"),
                            v.get("valid_to"),
                            v.get("condition_text"),
                            v["change_type"],
                            sup_vid,
                            smid,
                            v.get("source_quote") or v["summary"],
                            v.get("created_at") or _now_iso(),
                        ),
                    )
                    v_blob = None
                    if self.embedder is not None:
                        v_text = f"{it['title']} {v['summary']} {' '.join(ents)}".strip()
                        v_vecs = self.embedder.embed_passages([v_text])
                        if v_vecs and v_vecs[0]:
                            v_blob = pack_vector(v_vecs[0])
                    self._index_version_tx(
                        new_iid,
                        n_vid,
                        it["title"],
                        v["summary"],
                        v.get("effective_time_text"),
                        v.get("condition_text"),
                        ents,
                        v_blob,
                    )
            imported_items += 1

        for blk in payload.get("core_blocks") or ():
            self.upsert_core_memory_block(
                str(blk["block_key"]),
                str(blk["content"]),
                namespace=ns,
                source_item_ids=[],
                updated_at=blk.get("updated_at"),
            )
            imported_core += 1

        for ss in payload.get("session_summaries") or ():
            self.upsert_session_summary(
                str(ss["session_id"]),
                namespace=ns,
                session_date=str(ss["session_date"]),
                summary=str(ss["summary"]),
                facts=list(ss.get("facts") or ()),
                entities=list(ss.get("entities") or ()),
                source_message_ids=[
                    msg_id_remap.get(str(mid), str(mid))
                    for mid in (ss.get("source_message_ids") or ())
                ],
                updated_at=ss.get("updated_at"),
            )
            imported_summaries += 1

        if payload.get("consolidations"):
            self.replace_consolidations(
                [
                    {
                        "topic": c["topic"],
                        "summary": c["summary"],
                        "source_item_ids": [],
                        "source_message_ids": [],
                        "confidence": c.get("confidence", 0.85),
                    }
                    for c in payload["consolidations"]
                ],
                namespace=ns,
            )

        self._invalidate_cache(ns)
        return {
            "namespace": 1,
            "imported_messages": imported_msgs,
            "imported_items": imported_items,
            "imported_summaries": imported_summaries,
            "imported_core_blocks": imported_core,
        }

    @staticmethod
    def _row_to_message(row: sqlite3.Row) -> MessageRecord:
        return MessageRecord(
            id=row["id"],
            session_id=row["session_id"],
            namespace=row["namespace"] if "namespace" in row.keys() else "default",
            role=row["role"],
            content=row["content"],
            created_at=row["created_at"],
            timezone=row["timezone"],
            extracted=bool(row["extracted"]),
            deleted=bool(row["deleted"]),
            extraction_status=(
                "quarantined" if row["extraction_failure"] else "done" if row["extracted"] else "pending"
            ),
            extract_attempts=row["extract_attempts"],
            extraction_failure=row["extraction_failure"],
        )

    @staticmethod
    def _row_to_ambiguity(row: sqlite3.Row) -> AmbiguityRecord:
        return AmbiguityRecord(
            id=row["id"],
            namespace=row["namespace"] if "namespace" in row.keys() else "default",
            source_message_id=row["source_message_id"],
            source_quote=row["source_quote"],
            candidate_item_ids=tuple(json.loads(row["candidate_item_ids_json"])),
            note=row["note"],
            resolved=bool(row["resolved"]),
            created_at=row["created_at"],
        )
