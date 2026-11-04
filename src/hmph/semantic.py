"""Opt-in local semantic retrieval for public evaluation corpora.

No personal-memory disk cache is created. Product embeddings need the source
revision/epoch/deletion protocol before this ranker can be enabled online.
"""

import hashlib
import json
import math
import time
from collections import defaultdict
from importlib.metadata import version
from pathlib import Path


def reciprocal_rank_fusion(rankings, *, limit=80):
    if not 1 <= limit <= 80:
        raise ValueError("Invalid fusion limit")
    scores = defaultdict(float)
    for ranking in rankings:
        seen = set()
        for rank, candidate in enumerate(ranking, 1):
            if candidate not in seen:
                scores[candidate] += 1 / (60 + rank)
                seen.add(candidate)
    return sorted(scores, key=lambda i: (-scores[i], i))[:limit]


class LocalSemanticRanker:
    MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

    def __init__(self, cache_dir=Path("artifacts/models/multilingual-minilm")):
        from fastembed import TextEmbedding

        self.model = TextEmbedding(
            model_name=self.MODEL,
            cache_dir=str(cache_dir),
            threads=4,
            providers=["CPUExecutionProvider"],
        )
        fingerprints = {}
        for file in sorted(cache_dir.rglob("*")):
            if (
                file.is_file()
                and "snapshots" in file.parts
                and file.suffix in {".onnx", ".onnx_data", ".json", ".txt"}
            ):
                digest = hashlib.sha256()
                with file.open("rb") as source:
                    for data in iter(lambda: source.read(1024 * 1024), b""):
                        digest.update(data)
                fingerprints[str(file.relative_to(cache_dir)).replace("\\", "/")] = (
                    digest.hexdigest()
                )
        if not fingerprints:
            raise RuntimeError("Embedding artifact fingerprint unavailable")
        self.metadata = {
            "model": self.MODEL,
            "fastembed_version": version("fastembed"),
            "dimension": self.model.embedding_size,
            "device": "CPU",
            "license": "Apache-2.0",
            "artifact_sha256": fingerprints,
            "personal_data_cache": False,
            "production_enabled": False,
        }
        self._prepared = None

    @staticmethod
    def _corpus_key(documents):
        checksum = hashlib.sha256()
        for d in documents:
            identity = [
                d.source_id,
                d.session_id,
                d.timestamp,
                d.start_offset,
                d.end_offset,
                d.content,
            ]
            checksum.update(json.dumps(identity, ensure_ascii=False).encode())
            checksum.update(b"\n")
        return checksum.hexdigest()

    def prepare_public_index(self, documents):
        """Explicit public-only, in-memory index. Build cost remains reportable."""
        if not documents or len(documents) > 20000:
            raise ValueError("Public index requires 1..20000 passages")
        key = self._corpus_key(documents)
        started = time.monotonic()
        vectors = list(self.model.passage_embed([d.content for d in documents], batch_size=32))
        if len(vectors) != len(documents):
            raise RuntimeError("Incomplete embedding batch")
        self._prepared = (key, vectors)
        self.index_metrics = {
            "passages": len(documents),
            "build_seconds": time.monotonic() - started,
            "storage": "process_memory_only",
            "public_only": True,
        }

    def discard_public_index(self):
        self._prepared = None

    def rank(self, documents, question):
        if not documents:
            return []
        started = time.monotonic()
        query = list(next(self.model.query_embed(question)))
        query_seconds = time.monotonic() - started
        query_norm = math.sqrt(sum(float(v) * float(v) for v in query))
        if not query_norm:
            raise RuntimeError("Invalid query embedding")
        started = time.monotonic()
        prepared = getattr(self, "_prepared", None)
        if prepared is not None:
            if prepared[0] != self._corpus_key(documents):
                raise ValueError("Prepared public corpus changed; discard/rebuild explicitly")
            vectors = prepared[1]
        else:
            vectors = list(self.model.passage_embed([d.content for d in documents], batch_size=32))
        index_seconds = time.monotonic() - started
        started = time.monotonic()
        scores = []
        for index, vector in enumerate(vectors):
            norm = math.sqrt(sum(float(v) * float(v) for v in vector))
            if not norm or len(vector) != len(query):
                raise RuntimeError("Invalid passage embedding")
            score = sum(float(a) * float(b) for a, b in zip(query, vector, strict=True)) / (
                query_norm * norm
            )
            if not math.isfinite(score):
                raise RuntimeError("Invalid semantic score")
            scores.append((score, index))
        if len(scores) != len(documents):
            raise RuntimeError("Incomplete embedding batch")
        self.last_metrics = {
            "passages": len(documents),
            "query_seconds": query_seconds,
            "index_seconds": index_seconds,
            "scoring_seconds": time.monotonic() - started,
            "index_reused": prepared is not None,
        }
        return [index for _, index in sorted(scores, reverse=True)[:80]]
