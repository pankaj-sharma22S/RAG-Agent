"""Lightweight high-performance singleton embedding model powered by FastEmbed (ONNX Runtime).

Features:
- Pure CPU ONNX Runtime execution (zero heavy PyTorch / CUDA / graphics dependencies).
- Thread-safe in-memory caching wrapper.
- Automatic batching and embedding normalization.
"""

from __future__ import annotations

import hashlib
import logging
import os
from threading import Lock
from typing import Any, List
from langchain_core.embeddings import Embeddings

logger = logging.getLogger("EmbeddingModel")

try:
    from app.tracing import safe_traceable
except Exception:
    try:
        from tracing import safe_traceable
    except Exception:
        safe_traceable = lambda **kw: lambda f: f

DEFAULT_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "BAAI/bge-small-en-v1.5")

_base_embedding_model: Embeddings | None = None
_cached_embeddings_wrapper: CachedEmbeddings | None = None
_init_lock = Lock()
_init_count = 0
_first_init_ms = 0.0


class CachedEmbeddings(Embeddings):
    """Thread-safe in-memory caching wrapper around embedding models."""

    def __init__(self, base_embeddings: Embeddings, model_name: str = DEFAULT_MODEL_NAME, max_size: int = 10000) -> None:
        self.base = base_embeddings
        self.model_name = model_name
        self.max_size = max_size
        self._cache: dict[str, list[float]] = {}
        self._lock = Lock()
        self.hits = 0
        self.misses = 0

    def _hash(self, text: str) -> str:
        key = f"{text.strip()}|{self.model_name}|v1.5"
        return hashlib.sha256(key.encode("utf-8")).hexdigest()

    @safe_traceable(name="embed_query", run_type="embedding", tags=["embedding", "fastembed", "onnx"])
    def embed_query(self, text: str) -> list[float]:
        h = self._hash(text)
        with self._lock:
            if h in self._cache:
                self.hits += 1
                return self._cache[h]
            self.misses += 1

        vec = self.base.embed_query(text)
        with self._lock:
            if len(self._cache) >= self.max_size:
                keys = list(self._cache.keys())[:2000]
                for k in keys:
                    del self._cache[k]
            self._cache[h] = vec
        return vec

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        results: list[list[float] | None] = [None] * len(texts)
        uncached_indices: list[int] = []
        uncached_texts: list[str] = []

        with self._lock:
            for idx, text in enumerate(texts):
                h = self._hash(text)
                if h in self._cache:
                    self.hits += 1
                    results[idx] = self._cache[h]
                else:
                    self.misses += 1
                    uncached_indices.append(idx)
                    uncached_texts.append(text)

        if uncached_texts:
            new_vecs = self.base.embed_documents(uncached_texts)
            with self._lock:
                for idx, vec in zip(uncached_indices, new_vecs):
                    h = self._hash(texts[idx])
                    self._cache[h] = vec
                    results[idx] = vec

        return [r for r in results if r is not None]

    def stats(self) -> dict[str, Any]:
        with self._lock:
            total = self.hits + self.misses
            rate = (self.hits / total * 100.0) if total > 0 else 0.0
            return {
                "size": len(self._cache),
                "hits": self.hits,
                "misses": self.misses,
                "hit_rate_pct": round(rate, 2),
            }


def get_embedding_model() -> CachedEmbeddings:
    """Retrieve or lazily initialize the singleton lightweight cached embedding model."""
    global _base_embedding_model, _cached_embeddings_wrapper, _init_count, _first_init_ms

    if _cached_embeddings_wrapper is None:
        with _init_lock:
            if _cached_embeddings_wrapper is None:
                import time
                t0 = time.perf_counter()
                model_name = os.getenv("EMBEDDING_MODEL_NAME", DEFAULT_MODEL_NAME)
                logger.info(f"Initializing lightweight ONNX embedding model '{model_name}'...")

                try:
                    from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
                    _base_embedding_model = FastEmbedEmbeddings(model_name=model_name)
                    logger.info("FastEmbed ONNX engine loaded successfully (zero PyTorch dependency).")
                except Exception as err:
                    logger.warning("FastEmbed not available, attempting fallback: %s", err)
                    try:
                        from langchain_huggingface import HuggingFaceEmbeddings
                        _base_embedding_model = HuggingFaceEmbeddings(model_name=model_name)
                    except Exception as inner_err:
                        raise RuntimeError(f"Failed to initialize embedding model: {inner_err}") from inner_err

                _cached_embeddings_wrapper = CachedEmbeddings(_base_embedding_model, model_name=model_name)
                _first_init_ms = round((time.perf_counter() - t0) * 1000, 2)
                _init_count += 1
                try:
                    _cached_embeddings_wrapper.embed_query("warmup query")
                except Exception:
                    pass

    return _cached_embeddings_wrapper


def get_embedding_lifecycle_stats() -> dict[str, Any]:
    """Return initialization and caching metrics for the embedding engine."""
    cache_stats = _cached_embeddings_wrapper.stats() if _cached_embeddings_wrapper else {}
    return {
        "component": "EmbeddingModel",
        "engine": "FastEmbed_ONNX",
        "initialization_count": _init_count,
        "first_init_ms": _first_init_ms,
        "cache": cache_stats,
    }
