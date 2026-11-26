"""Heterogeneous Entity-Item-Summary-Message graph and Personalized PageRank (HippoRAG-2 inspired).

Zero external graph DB required: builds an in-memory sparse adjacency graph directly from
SQLite items, entities, versions, coreference-resolved session summaries (passage nodes),
and Zettelkasten memory_edges in milliseconds, weights entity bridges by inverse degree (IDF
specificity), and runs sparse Personalized PageRank to surface multi-hop cross-session evidence
with zero LLM cost.
"""

from __future__ import annotations

import json
import math
import re
import sqlite3
from collections import defaultdict

from hmph.lexical import tokenize

_CLEAN_ENTITY = re.compile(r"[^\w\u4e00-\u9fff]+")
_GENERIC_STOP = {
    "user",
    "assistant",
    "today",
    "yesterday",
    "tomorrow",
    "week",
    "month",
    "year",
    "time",
    "work",
    "home",
    "life",
    "new",
    "plan",
    "plans",
    "day",
    "night",
    "weekend",
    "things",
    "something",
    "anything",
}


def normalize_entity(raw: str) -> str:
    cleaned = _CLEAN_ENTITY.sub(" ", raw.strip().lower()).strip()
    if len(cleaned) < 2 or cleaned in _GENERIC_STOP:
        return ""
    return cleaned


class MemoryBipartiteGraph:
    """HippoRAG-2 style sparse heterogeneous graph over entities, items, summaries, and messages."""

    def __init__(self) -> None:
        # node_key -> list of (neighbor_key, weight)
        self.adj: dict[str, list[tuple[str, float]]] = defaultdict(list)
        self.entity_nodes: set[str] = set()
        self.entity_doc_freq: dict[str, int] = defaultdict(int)
        self.item_entities: dict[str, set[str]] = defaultdict(set)

    def _add_undirected(self, a: str, b: str, weight: float) -> None:
        if a == b or weight <= 0.0:
            return
        self.adj[a].append((b, weight))
        self.adj[b].append((a, weight))

    def _apply_entity_idf_scaling(self) -> None:
        """Dampen ubiquitous hub entities so specific names/topics carry sharper PPR mass."""
        if not self.entity_doc_freq:
            return
        scale_by_ent: dict[str, float] = {}
        for norm, df in self.entity_doc_freq.items():
            scale_by_ent[f"ent:{norm}"] = 1.0 / (1.0 + 0.30 * math.log1p(max(0, df - 1)))

        for node, neighbors in list(self.adj.items()):
            if node in scale_by_ent:
                s = scale_by_ent[node]
                self.adj[node] = [(nb, w * s) for nb, w in neighbors]
            else:
                self.adj[node] = [
                    (nb, w * scale_by_ent[nb]) if nb in scale_by_ent else (nb, w)
                    for nb, w in neighbors
                ]

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
            raw_entities = list(ents)
            for match in re.findall(
                r"[A-Z][a-zA-Z0-9-]+(?:\s+[A-Z][a-zA-Z0-9-]+){0,3}", row["title"]
            ):
                raw_entities.append(match)
            for tok in tokenize(row["title"]):
                if len(tok) >= 2:
                    raw_entities.append(tok)

            seen_ents: set[str] = set()
            explicit_norms = {normalize_entity(str(e)) for e in ents}
            for ent in raw_entities:
                norm = normalize_entity(str(ent))
                if not norm or norm in seen_ents:
                    continue
                seen_ents.add(norm)
                ent_key = f"ent:{norm}"
                graph.entity_nodes.add(norm)
                graph.entity_doc_freq[norm] += 1
                graph.item_entities[row["id"]].add(norm)
                weight = 1.45 if norm in explicit_norms else 0.70
                graph._add_undirected(item_key, ent_key, weight)

        # 2. Connect item versions to their authoritative source messages & session nodes.
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
            sess_key = f"sess:{row['session_id']}"
            graph._add_undirected(msg_key, sess_key, 0.35)

        # 3. HippoRAG-2 Passage Integration: include coreference-resolved session_summaries (`sum:`)
        # so unextracted conversational episodes participate in multi-hop associative PPR!
        try:
            ss_rows = conn.execute(
                """
                SELECT id, session_id, entities_json, source_message_ids_json
                FROM session_summaries
                WHERE namespace = ?
                """,
                (namespace,),
            ).fetchall()
        except sqlite3.OperationalError:
            ss_rows = []

        for srow in ss_rows:
            sum_key = f"sum:{srow['id']}"
            sess_key = f"sess:{srow['session_id']}"
            graph._add_undirected(sum_key, sess_key, 0.55)
            seen_s_ents: set[str] = set()
            for ent in json.loads(srow["entities_json"] or "[]"):
                norm = normalize_entity(str(ent))
                if not norm or norm in seen_s_ents:
                    continue
                seen_s_ents.add(norm)
                graph.entity_nodes.add(norm)
                graph.entity_doc_freq[norm] += 1
                graph._add_undirected(sum_key, f"ent:{norm}", 1.25)
            for mid in json.loads(srow["source_message_ids_json"] or "[]")[:12]:
                if mid:
                    graph._add_undirected(sum_key, f"msg:{mid}", 0.85)

        # 4. Connect explicit semantic/Zettelkasten & evolution edges between items in this namespace.
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

        graph._apply_entity_idf_scaling()
        return graph

    def query_seed_entities(self, query: str) -> list[tuple[str, float]]:
        """Return matched entity nodes paired with IDF specificity boost."""
        q_low = query.lower()
        q_tokens = set(tokenize(query))
        matched: list[tuple[str, float]] = []
        for ent in self.entity_nodes:
            hit = False
            if len(ent) >= 3 and ent in q_low:
                hit = True
            elif ent in q_tokens or (ent.isascii() and f"{ent}s" in q_tokens):
                hit = True
            if hit:
                df = self.entity_doc_freq.get(ent, 1)
                idf_w = 2.2 / (1.0 + 0.25 * math.log1p(max(0, df - 1)))
                matched.append((ent, idf_w))
        matched.sort(key=lambda pair: (-pair[1], -len(pair[0]), pair[0]))
        return matched[:24]

    def personalized_pagerank_full(
        self,
        *,
        query: str,
        seed_item_ids: list[str],
        seed_message_ids: list[str],
        seed_summary_ids: list[str] | tuple[str, ...] = (),
        alpha: float = 0.25,
        iterations: int = 6,
        top_items: int = 16,
        top_messages: int = 20,
        top_summaries: int = 8,
    ) -> tuple[list[tuple[str, float]], list[tuple[str, float]], list[tuple[str, float]]]:
        """Run sparse Personalized PageRank returning ranked (items, messages, session_summaries)."""
        personalization: dict[str, float] = defaultdict(float)

        for ent, weight in self.query_seed_entities(query):
            personalization[f"ent:{ent}"] += weight
        for rank, iid in enumerate(seed_item_ids[:10], start=1):
            personalization[f"item:{iid}"] += 1.5 / (1.0 + 0.2 * rank)
        for rank, sid in enumerate(seed_summary_ids[:8], start=1):
            personalization[f"sum:{sid}"] += 1.35 / (1.0 + 0.2 * rank)
        for rank, mid in enumerate(seed_message_ids[:12], start=1):
            personalization[f"msg:{mid}"] += 1.0 / (1.0 + 0.2 * rank)

        if not personalization:
            return [], [], []

        total_p = sum(personalization.values())
        p_vec = {node: val / total_p for node, val in personalization.items()}
        scores = dict(p_vec)

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
        sum_ranked = sorted(
            ((k[4:], v) for k, v in scores.items() if k.startswith("sum:")),
            key=lambda pair: (-pair[1], pair[0]),
        )[:top_summaries]
        return item_ranked, msg_ranked, sum_ranked

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
        """Backward-compatible 2-tuple wrapper returning (items, messages)."""
        items, msgs, _ = self.personalized_pagerank_full(
            query=query,
            seed_item_ids=seed_item_ids,
            seed_message_ids=seed_message_ids,
            alpha=alpha,
            iterations=iterations,
            top_items=top_items,
            top_messages=top_messages,
        )
        return items, msgs
