"""Pluggable embedding backends for exact local semantic retrieval in SQLite."""

from __future__ import annotations

import hashlib
import math
import os
import struct
from pathlib import Path
from typing import Protocol

from hmph.lexical import tokenize


class Embedder(Protocol):
    name: str
    dimension: int

    def embed_passages(self, texts: list[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


def pack_vector(vector: list[float]) -> bytes:
    return struct.pack(f"<{len(vector)}f", *vector)


def unpack_vector(blob: bytes) -> list[float]:
    count = len(blob) // 4
    return list(struct.unpack(f"<{count}f", blob))


def cosine_similarity(left: list[float], right: list[float]) -> float:
    if len(left) != len(right) or not left:
        return 0.0
    dot = sum(a * b for a, b in zip(left, right, strict=True))
    norm_l = math.sqrt(sum(a * a for a in left))
    norm_r = math.sqrt(sum(b * b for b in right))
    if norm_l == 0.0 or norm_r == 0.0:
        return 0.0
    score = dot / (norm_l * norm_r)
    return score if math.isfinite(score) else 0.0


def batch_cosine_top_k(
    query: list[float],
    candidates: list[tuple[str, bytes]],
    *,
    top_k: int = 40,
    min_score: float = 0.05,
) -> list[tuple[str, float]]:
    if not query or not candidates:
        return []
    try:
        import numpy as np

        q = np.asarray(query, dtype=np.float32)
        q_norm = float(np.linalg.norm(q))
        if q_norm == 0.0:
            return []
        dim = len(query)
        valid_ids: list[str] = []
        valid_rows: list[np.ndarray] = []
        for cid, blob in candidates:
            if len(blob) == dim * 4:
                valid_ids.append(cid)
                valid_rows.append(np.frombuffer(blob, dtype=np.float32))
        if not valid_rows:
            return []
        mat = np.vstack(valid_rows)
        norms = np.linalg.norm(mat, axis=1)
        denom = norms * q_norm
        scores = np.where(denom > 0, (mat @ q) / denom, 0.0)
        order = np.argsort(-scores)
        results: list[tuple[str, float]] = []
        for idx in order[:top_k]:
            score = float(scores[idx])
            if math.isfinite(score) and score >= min_score:
                results.append((valid_ids[int(idx)], score))
        return results
    except ImportError:
        scored = [
            (cid, cosine_similarity(query, unpack_vector(blob)))
            for cid, blob in candidates
        ]
        scored = [(cid, s) for cid, s in scored if s >= min_score]
        scored.sort(key=lambda pair: (-pair[1], pair[0]))
        return scored[:top_k]


def build_normalized_matrix(
    candidates: list[tuple[str, bytes]], dim: int
) -> tuple[list[str], object]:
    """Pre-stack and L2-normalize candidate float32 blobs for sub-millisecond cached search."""
    if not candidates or dim <= 0:
        return [], None
    expected_bytes = dim * 4
    try:
        import numpy as np

        valid_ids: list[str] = []
        valid_rows: list[np.ndarray] = []
        for cid, blob in candidates:
            if len(blob) == expected_bytes:
                valid_ids.append(cid)
                valid_rows.append(np.frombuffer(blob, dtype=np.float32))
        if not valid_rows:
            return [], None
        mat = np.vstack(valid_rows).astype(np.float32, copy=True)
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        mat = np.where(norms > 1e-9, mat / np.clip(norms, 1e-9, None), 0.0)
        return valid_ids, mat
    except ImportError:
        unpacked = [
            (cid, unpack_vector(blob))
            for cid, blob in candidates
            if len(blob) == expected_bytes
        ]
        return [cid for cid, _ in unpacked], [vec for _, vec in unpacked]


def matrix_cosine_top_k(
    query: list[float],
    valid_ids: list[str],
    norm_matrix: object,
    *,
    top_k: int = 40,
    min_score: float = 0.05,
) -> list[tuple[str, float]]:
    """Query a cached L2-normalized Float32 matrix in <1ms even across 10k+ vectors."""
    if not query or not valid_ids or norm_matrix is None:
        return []
    try:
        import numpy as np

        if isinstance(norm_matrix, np.ndarray):
            q = np.asarray(query, dtype=np.float32)
            q_norm = float(np.linalg.norm(q))
            if q_norm <= 1e-9 or norm_matrix.shape[1] != q.shape[0]:
                return []
            scores = norm_matrix @ (q / q_norm)
            k = min(top_k, len(valid_ids))
            if k < len(valid_ids) and len(valid_ids) > 64:
                part = np.argpartition(-scores, k - 1)[:k]
                order = part[np.argsort(-scores[part])]
            else:
                order = np.argsort(-scores)[:k]
            results: list[tuple[str, float]] = []
            for idx in order:
                sc = float(scores[idx])
                if math.isfinite(sc) and sc >= min_score:
                    results.append((valid_ids[int(idx)], sc))
            return results
    except ImportError:
        pass

    if isinstance(norm_matrix, list):
        scored = [
            (cid, cosine_similarity(query, vec))
            for cid, vec in zip(valid_ids, norm_matrix, strict=False)
        ]
        scored = [(cid, s) for cid, s in scored if s >= min_score]
        scored.sort(key=lambda pair: (-pair[1], pair[0]))
        return scored[:top_k]
    return []


class LocalFastEmbedder:
    """Offline multilingual ONNX embedder reusing cached local MiniLM artifacts."""

    MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

    def __init__(self, cache_dir: Path | str = Path("artifacts/models/multilingual-minilm")):
        cache_path = Path(cache_dir)
        self._mode = "fastembed"
        self._ort_session = None
        self._tokenizer = None
        self._model = None

        # Prefer direct onnxruntime + tokenizers (avoids Pillow DLL dependency in fastembed).
        onnx_file = next(cache_path.rglob("*.onnx"), None) if cache_path.exists() else None
        tok_file = (
            onnx_file.parent / "tokenizer.json"
            if onnx_file is not None and (onnx_file.parent / "tokenizer.json").exists()
            else None
        )
        if onnx_file is not None and tok_file is not None:
            try:
                import onnxruntime as ort
                from tokenizers import Tokenizer

                self._ort_session = ort.InferenceSession(
                    str(onnx_file), providers=["CPUExecutionProvider"]
                )
                self._tokenizer = Tokenizer.from_file(str(tok_file))
                self._tokenizer.enable_truncation(max_length=256)
                self._tokenizer.enable_padding(pad_id=0, pad_token="<pad>")
                self._mode = "onnxruntime"
                self.name = f"onnx:{self.MODEL}"
                self.dimension = 384
                return
            except Exception:
                pass

        from fastembed import TextEmbedding

        kwargs: dict[str, object] = {
            "model_name": self.MODEL,
            "threads": 4,
            "providers": ["CPUExecutionProvider"],
        }
        if cache_path.exists():
            kwargs["cache_dir"] = str(cache_path)
            kwargs["local_files_only"] = True
        self._model = TextEmbedding(**kwargs)
        self.name = f"fastembed:{self.MODEL}"
        self.dimension = int(self._model.embedding_size)

    def _embed_onnx(self, texts: list[str]) -> list[list[float]]:
        import numpy as np

        assert self._tokenizer is not None and self._ort_session is not None
        encoded = self._tokenizer.encode_batch(texts)
        input_ids = np.asarray([e.ids for e in encoded], dtype=np.int64)
        attention_mask = np.asarray([e.attention_mask for e in encoded], dtype=np.int64)
        inputs = {"input_ids": input_ids, "attention_mask": attention_mask}
        input_names = {inp.name for inp in self._ort_session.get_inputs()}
        if "token_type_ids" in input_names:
            inputs["token_type_ids"] = np.zeros_like(input_ids)
        token_embeddings = self._ort_session.run(None, inputs)[0]
        mask = attention_mask[:, :, None].astype(np.float32)
        pooled = (token_embeddings * mask).sum(axis=1) / np.clip(mask.sum(axis=1), 1e-9, None)
        norms = np.linalg.norm(pooled, axis=1, keepdims=True)
        normalized = pooled / np.clip(norms, 1e-9, None)
        return [row.tolist() for row in normalized]

    def embed_passages(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        if self._mode == "onnxruntime":
            return self._embed_onnx(texts)
        assert self._model is not None
        return [list(map(float, vec)) for vec in self._model.passage_embed(texts, batch_size=32)]

    def embed_query(self, text: str) -> list[float]:
        if self._mode == "onnxruntime":
            return self._embed_onnx([text])[0]
        assert self._model is not None
        return list(map(float, next(self._model.query_embed(text))))


class OpenAIEmbedder:
    """Optional cloud/local OpenAI-compatible embedding backend for strong benchmarks."""

    def __init__(self, api_key: str, *, base_url: str, model: str):
        import httpx

        self._client = httpx.Client(
            base_url=base_url.rstrip("/"),
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=httpx.Timeout(20.0, connect=10.0),
        )
        self._model = model
        self.name = f"openai-embed:{model}"
        self.dimension = 0

    def _request(self, texts: list[str]) -> list[list[float]]:
        response = self._client.post("/embeddings", json={"model": self._model, "input": texts})
        response.raise_for_status()
        data = response.json()["data"]
        data.sort(key=lambda item: item["index"])
        vectors = [[float(x) for x in item["embedding"]] for item in data]
        if vectors and not self.dimension:
            self.dimension = len(vectors[0])
        return vectors

    def embed_passages(self, texts: list[str]) -> list[list[float]]:
        return self._request(texts) if texts else []

    def embed_query(self, text: str) -> list[float]:
        return self._request([text])[0]

    def close(self) -> None:
        self._client.close()


class HashingEmbedder:
    """Fast deterministic n-gram hashing embedder for zero-cost offline tests."""

    def __init__(self, dimension: int = 128):
        self.dimension = dimension
        self.name = f"hashing:{dimension}"

    def _encode(self, text: str) -> list[float]:
        vec = [0.0] * self.dimension
        tokens = tokenize(text)
        if not tokens:
            return vec
        for token in tokens:
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=4).digest()
            slot = int.from_bytes(digest, "little") % self.dimension
            vec[slot] += 1.0
        norm = math.sqrt(sum(v * v for v in vec))
        return [v / norm for v in vec] if norm > 0 else vec

    def embed_passages(self, texts: list[str]) -> list[list[float]]:
        return [self._encode(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._encode(text)


_ONNX_PROBE_CACHE: bool | None = None


def _can_import_in_subprocess(module_name: str) -> bool:
    """Probe native DLL imports in a child process so a hard DLL crash never kills the host."""
    import subprocess
    import sys

    try:
        proc = subprocess.run(
            [sys.executable, "-c", f"import {module_name}"],
            capture_output=True,
            timeout=10,
            check=False,
        )
        return proc.returncode == 0
    except Exception:
        return False


def build_default_embedder(
    cache_dir: Path | str = Path("artifacts/models/multilingual-minilm"),
) -> Embedder:
    """Select cloud OpenAI embedder if configured, else cached local ONNX, else hashing."""
    global _ONNX_PROBE_CACHE
    api_key = os.getenv("HMPH_EMBEDDING_API_KEY", "").strip()
    base_url = os.getenv("HMPH_EMBEDDING_BASE_URL", "").strip()
    model = os.getenv("HMPH_EMBEDDING_MODEL", "").strip()
    if api_key and base_url and model:
        return OpenAIEmbedder(api_key, base_url=base_url, model=model)
    if os.getenv("HMPH_DISABLE_LOCAL_ONNX", "").strip() != "1":
        if _ONNX_PROBE_CACHE is None:
            _ONNX_PROBE_CACHE = _can_import_in_subprocess("onnxruntime")
        if _ONNX_PROBE_CACHE:
            try:
                return LocalFastEmbedder(cache_dir=cache_dir)
            except Exception:
                pass
    return HashingEmbedder()
