"""Neural Cross-Encoder reranker singleton with score caching, batched inference, and CPU-optimized execution."""

from __future__ import annotations

import hashlib
import logging
import os
from threading import Lock
from typing import Any, List, Tuple
from langchain_core.documents import Document

logger = logging.getLogger("CrossEncoderReranker")


try:
    from app.tracing import safe_traceable
except Exception:
    try:
        from tracing import safe_traceable
    except Exception:
        safe_traceable = lambda **kw: lambda f: f


class CrossEncoderReranker:
    """Thread-safe, cached Cross-Encoder reranker for top candidate refinement."""

    def __init__(self) -> None:
        self.model_name = os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
        self.rerank_top_k = int(os.getenv("RERANK_TOP_K", "6"))
        self.max_candidates = int(os.getenv("RERANK_MAX_CANDIDATES", "8"))
        self._model = None
        self._score_cache: dict[str, float] = {}
        self._lock = Lock()
        self._disabled = os.getenv("RERANKER_ENABLED", "true").lower() in {"false", "0", "no"}
        self.init_count = 0
        self.first_init_ms = 0.0
        self.hits = 0
        self.misses = 0

    def _get_model(self) -> Any:
        """Lazy thread-safe initialization of the Cross-Encoder model."""
        if self._disabled:
            return None

        if self._model is None:
            with self._lock:
                if self._model is None:
                    import time
                    t0 = time.perf_counter()
                    try:
                        from sentence_transformers import CrossEncoder

                        logger.info(f"Loading CrossEncoder model: {self.model_name} (CPU inference)")
                        self._model = CrossEncoder(self.model_name)
                        self.first_init_ms = round((time.perf_counter() - t0) * 1000, 2)
                        self.init_count += 1
                    except Exception as err:
                        logger.warning(
                            f"Failed to load primary CrossEncoder '{self.model_name}': {err}. "
                            "Attempting lightweight fallback 'cross-encoder/ms-marco-MiniLM-L-6-v2'..."
                        )
                        try:
                            from sentence_transformers import CrossEncoder

                            self._model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
                            self.first_init_ms = round((time.perf_counter() - t0) * 1000, 2)
                            self.init_count += 1
                        except Exception as inner_err:
                            logger.error(
                                f"CrossEncoder initialization failed: {inner_err}. "
                                "Falling back to rank preservation."
                            )
                            self._disabled = True
                            self._model = None

        return self._model

    def _pair_hash(self, query: str, text: str) -> str:
        h = hashlib.md5()
        h.update(query.strip().lower().encode("utf-8"))
        h.update(text[:500].encode("utf-8"))
        return h.hexdigest()

    @safe_traceable(name="cross_encoder_rerank", run_type="chain", tags=["reranker", "cross-encoder", "relevance"])
    def rerank(
        self,
        query: str,
        documents: List[Document] | List[Tuple[Document, float]],
        top_k: int | None = None,
    ) -> List[Tuple[Document, float, int]]:
        """Rerank candidates using neural Cross-Encoder in batches with score caching."""
        k = top_k if top_k is not None else self.rerank_top_k
        if not documents or not query.strip():
            return []

        doc_list: List[Document] = []
        for item in documents:
            if isinstance(item, tuple):
                doc_list.append(item[0])
            else:
                doc_list.append(item)

        if not doc_list:
            return []

        doc_list = doc_list[: self.max_candidates]
        model = self._get_model()

        if model is None:
            return [(doc, 1.0 - (idx * 0.05), idx + 1) for idx, doc in enumerate(doc_list[:k])]

        try:
            scored_docs: List[Tuple[Document, float]] = []
            uncached_pairs: List[List[str]] = []
            uncached_indices: List[int] = []

            with self._lock:
                for idx, doc in enumerate(doc_list):
                    phash = self._pair_hash(query, doc.page_content)
                    if phash in self._score_cache:
                        self.hits += 1
                        scored_docs.append((doc, self._score_cache[phash]))
                    else:
                        self.misses += 1
                        uncached_pairs.append([query, doc.page_content[:500]])
                        uncached_indices.append(idx)

            if uncached_pairs:
                scores = model.predict(uncached_pairs)
                with self._lock:
                    for i, idx in enumerate(uncached_indices):
                        doc = doc_list[idx]
                        raw_score = float(scores[i]) if hasattr(scores, "__getitem__") else float(scores)
                        # Normalize logit to [0.0, 1.0] probability via sigmoid
                        try:
                            import math
                            prob = 1.0 / (1.0 + math.exp(-raw_score))
                        except Exception:
                            prob = 0.0 if raw_score < 0 else 1.0
                        calibrated = round(prob, 4)
                        phash = self._pair_hash(query, doc.page_content)
                        self._score_cache[phash] = calibrated
                        scored_docs.append((doc, calibrated))

            scored_docs.sort(key=lambda x: x[1], reverse=True)

            results: List[Tuple[Document, float, int]] = []
            for rank, (doc, score) in enumerate(scored_docs[:k], start=1):
                results.append((doc, score, rank))

            return results
        except Exception as err:
            logger.warning(f"Cross-Encoder reranking error: {err}. Preserving fusion rank order.")
            return [(doc, 1.0 - (idx * 0.05), idx + 1) for idx, doc in enumerate(doc_list[:k])]

    def stats(self) -> dict[str, Any]:
        with self._lock:
            total = self.hits + self.misses
            rate = (self.hits / total * 100.0) if total > 0 else 0.0
            return {
                "component": "CrossEncoder",
                "initialization_count": self.init_count,
                "first_init_ms": self.first_init_ms,
                "cache_size": len(self._score_cache),
                "hits": self.hits,
                "misses": self.misses,
                "hit_rate_pct": round(rate, 2),
            }


global_reranker = CrossEncoderReranker()


def get_reranker() -> CrossEncoderReranker:
    """Return the global CrossEncoderReranker singleton."""
    return global_reranker


def get_cross_encoder():
    """Return the underlying CrossEncoder model instance."""
    return global_reranker._get_model()


def rerank_documents(
    query: str,
    documents: List[Document] | List[Tuple[Document, float]],
    top_k: int | None = None,
) -> List[Document]:
    """Convenience function to rerank documents and return enriched Document instances."""
    results = global_reranker.rerank(query=query, documents=documents, top_k=top_k)
    enriched = []
    for doc, score, rank in results:
        doc.metadata["rerank_score"] = score
        doc.metadata["rerank_rank"] = rank
        enriched.append(doc)
    return enriched


__all__ = [
    "CrossEncoderReranker",
    "global_reranker",
    "get_reranker",
    "get_cross_encoder",
    "rerank_documents",
]
