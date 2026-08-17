"""High-performance singleton embedding model with in-memory caching and optimized batching."""

from __future__ import annotations

import hashlib
import logging
import os
import torch
from threading import Lock
from typing import Any
from langchain_core.embeddings import Embeddings
try:
    from langchain_huggingface import HuggingFaceEmbeddings
except ModuleNotFoundError:
    try:
        from langchain_community.embeddings import HuggingFaceEmbeddings
    except ModuleNotFoundError as err:
        raise ModuleNotFoundError(
            "Neither 'langchain-huggingface' nor 'langchain-community' is installed. "
            "Please install langchain-huggingface using: pip install langchain-huggingface"
        ) from err

logger = logging.getLogger("EmbeddingModel")

# Configure torch threads for maximum CPU throughput
try:
    if not torch.cuda.is_available():
        torch.set_num_threads(min(8, os.cpu_count() or 4))
except Exception:
    pass

DEFAULT_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "BAAI/bge-small-en-v1.5")

_base_embedding_model: HuggingFaceEmbeddings | None = None
_cached_embeddings_wrapper: CachedEmbeddings | None = None
_init_lock = Lock()
_init_count = 0
_first_init_ms = 0.0


class CachedEmbeddings(Embeddings):
    """Thread-safe in-memory caching wrapper around HuggingFaceEmbeddings."""

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
                # Evict 20% oldest entries
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
    """Retrieve or lazily initialize the singleton cached embedding model."""
    global _base_embedding_model, _cached_embeddings_wrapper, _init_count, _first_init_ms

    if _cached_embeddings_wrapper is None:
        with _init_lock:
            if _cached_embeddings_wrapper is None:
                import time
                t0 = time.perf_counter()
                model_name = os.getenv("EMBEDDING_MODEL_NAME", DEFAULT_MODEL_NAME)
                logger.info(f"Initializing embedding model '{model_name}'...")
                device = "cuda" if torch.cuda.is_available() else "cpu"
                _base_embedding_model = HuggingFaceEmbeddings(
                    model_name=model_name,
                    model_kwargs={"device": device},
                    encode_kwargs={"batch_size": 64, "normalize_embeddings": True},
                )
                _cached_embeddings_wrapper = CachedEmbeddings(_base_embedding_model, model_name=model_name)
                _first_init_ms = round((time.perf_counter() - t0) * 1000, 2)
                _init_count += 1
                # Warm-up call
                try:
                    _cached_embeddings_wrapper.embed_query("warmup query")
                except Exception:
                    pass
                logger.info(f"Embedding model '{model_name}' initialized on {device} with in-memory caching.")

    return _cached_embeddings_wrapper


def get_embedding_lifecycle_stats() -> dict[str, Any]:
    """Return initialization and caching metrics for the embedding engine."""
    cache_stats = _cached_embeddings_wrapper.stats() if _cached_embeddings_wrapper else {}
    return {
        "component": "EmbeddingModel",
        "initialization_count": _init_count,
        "first_init_ms": _first_init_ms,
        "cache": cache_stats,
    }
