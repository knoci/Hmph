"""Entity-Memory-Message bipartite graph and Personalized PageRank (HippoRAG-inspired).

Zero external graph DB required: builds an in-memory sparse adjacency graph directly from
SQLite items, entities, versions, and memory_edges in milliseconds, then runs 2-step PPR
to surface long-tail cross-session episodes and multi-hop evidence with zero LLM cost.
"""

from __future__ import annotations

import json
import re
import sqlite3
from collections import defaultdict

from hmph.lexical import tokenize

_CLEAN_ENTITY = re.compile(r"[^\w\u4e00-\u9fff]+")


def normalize_entity(raw: str) -> str:
    cleaned = _CLEAN_ENTITY.sub(" ", raw.strip().lower()).strip()
    return cleaned if len(cleaned) >= 2 else ""


class MemoryBipartiteGraph:
    """Lightweight sparse graph over entities, items, and raw messages."""

    def __init__(self) -> None:
        # node_key -> list of (neighbor_key, weight)
        self.adj: dict[str, list[tuple[str, float]]] = defaultdict(list)
        self.entity_nodes: set[str] = set()
        self.item_entities: dict[str, set[str]] = defaultdict(set)

    def _add_undirected(self, a: str, b: str, weight: float) -> None:
        if a == b:
            return
        self.adj[a].append((b, weight))
        self.adj[b].append((a, weight))

    @classmethod
    def from_sqlite(
        cls, conn: sqlite3.Connection, *, namespace: str = "default"
    ) -> MemoryBipartiteGraph:
        graph = cls()

        # 1. Connect items to normalized entities and title tokens within this namespace.
        item_rows = conn.execute(
            "SELECT id, title, kind, status, entities_json FROM items WHERE namespace = ?",
            (namespace,),
        ).fetchall()
        item_ids = {r["id"] for r in item_rows}
        for row in item_rows:
            item_key = f"item:{row['id']}"
            ents = json.loads(row["entities_json"] or "[]")
            # Also add kind as a weak category bridge (e.g. episode/plan) and salient title keywords.
            raw_entities = list(ents)
            # Extract English capitalized phrases & Chinese words from title as implicit entities.
            for match in re.findall(r"[A-Z][a-zA-Z0-9-]+(?:\s+[A-Z][a-zA-Z0-9-]+){0,3}", row["title"]):
                raw_entities.append(match)
            for tok in tokenize(row["title"]):
                if len(tok) >= 2:
                    raw_entities.append(tok)

            seen_ents: set[str] = set()
            for ent in raw_entities:
                norm = normalize_entity(str(ent))
                if not norm or norm in seen_ents:
                    continue
                seen_ents.add(norm)
                ent_key = f"ent:{norm}"
                graph.entity_nodes.add(norm)
                graph.item_entities[row["id"]].add(norm)
                # Explicit extracted entities get higher weight than generic title tokens.
                weight = 1.4 if ent in ents else 0.7
                graph._add_undirected(item_key, ent_key, weight)

        # 2. Connect item versions to their authoritative source messages in this namespace.
        ver_rows = conn.execute(
            """
            SELECT v.item_id, v.source_message_id, v.status, m.session_id
            FROM item_versions AS v
            JOIN messages AS m ON m.id = v.source_message_id
            JOIN items AS i ON i.id = v.item_id
            WHERE m.deleted = 0 AND m.namespace = ? AND i.namespace = ?
            """,
            (namespace, namespace),
        ).fetchall()
        for row in ver_rows:
            item_key = f"item:{row['item_id']}"
            msg_key = f"msg:{row['source_message_id']}"
            graph._add_undirected(item_key, msg_key, 1.2)
            # Lightly connect item to its session node for cross-turn coherence.
            sess_key = f"sess:{row['session_id']}"
            graph._add_undirected(msg_key, sess_key, 0.35)

        # 3. Connect explicit semantic & evolution edges between items in this namespace.
        edge_rows = conn.execute(
            """
            SELECT src_type, src_id, dst_type, dst_id, relation
            FROM memory_edges
            WHERE src_type = 'item' AND dst_type = 'item'
            """
        ).fetchall()
        for row in edge_rows:
            if row["src_id"] not in item_ids or row["dst_id"] not in item_ids:
                continue
            rel = row["relation"]
            weight = 1.5 if rel in {"RELATED_ENTITY", "SUPPORTS"} else 1.0
            graph._add_undirected(f"item:{row['src_id']}", f"item:{row['dst_id']}", weight)

        return graph

    def query_seed_entities(self, query: str) -> list[str]:
        q_low = query.lower()
        seeds: list[str] = []
        q_tokens = set(tokenize(query))
        for ent in self.entity_nodes:
            if len(ent) >= 3 and ent in q_low:
                seeds.append(ent)
            elif ent in q_tokens:
                seeds.append(ent)
        return seeds[:20]

    def personalized_pagerank(
        self,
        *,
        query: str,
        seed_item_ids: list[str],
        seed_message_ids: list[str],
        alpha: float = 0.25,
        iterations: int = 6,
        top_items: int = 16,
        top_messages: int = 20,
    ) -> tuple[list[tuple[str, float]], list[tuple[str, float]]]:
        """Run sparse Personalized PageRank from query entities + first-pass seeds."""
        personalization: dict[str, float] = defaultdict(float)

        for ent in self.query_seed_entities(query):
            personalization[f"ent:{ent}"] += 2.0
        for rank, iid in enumerate(seed_item_ids[:10], start=1):
            personalization[f"item:{iid}"] += 1.5 / (1.0 + 0.2 * rank)
        for rank, mid in enumerate(seed_message_ids[:12], start=1):
            personalization[f"msg:{mid}"] += 1.0 / (1.0 + 0.2 * rank)

        if not personalization:
            return [], []

        total_p = sum(personalization.values())
        p_vec = {node: val / total_p for node, val in personalization.items()}
        scores = dict(p_vec)

        # Precompute outgoing weight sums for fast sparse iteration.
        out_sums = {
            node: sum(w for _, w in neighbors) or 1.0 for node, neighbors in self.adj.items()
        }

        for _ in range(iterations):
            next_scores: dict[str, float] = defaultdict(float)
            for node, base in p_vec.items():
                next_scores[node] += alpha * base
            for node, sc in scores.items():
                neighbors = self.adj.get(node)
                if not neighbors or sc < 1e-7:
                    continue
                denom = out_sums[node]
                mass = (1.0 - alpha) * sc
                for nb, w in neighbors:
                    next_scores[nb] += mass * (w / denom)
            scores = next_scores

        item_ranked = sorted(
            ((k[5:], v) for k, v in scores.items() if k.startswith("item:")),
            key=lambda pair: (-pair[1], pair[0]),
        )[:top_items]
        msg_ranked = sorted(
            ((k[4:], v) for k, v in scores.items() if k.startswith("msg:")),
            key=lambda pair: (-pair[1], pair[0]),
        )[:top_messages]
        return item_ranked, msg_ranked
