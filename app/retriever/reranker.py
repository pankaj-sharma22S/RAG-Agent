"""Cross-Encoder reranker singleton with batched inference and score caching."""

from __future__ import annotations

import hashlib
import logging
import os
from threading import Lock
from typing import Any
from langchain_core.documents import Document

logger = logging.getLogger("CrossEncoderReranker")


class CrossEncoderReranker:
    """Thread-safe, cached Cross-Encoder reranker for top candidate refinement."""

    def __init__(self) -> None:
        self.model_name = os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
        self.rerank_top_k = int(os.getenv("RERANK_TOP_K", "8"))
        self._model = None
        self._score_cache: dict[str, float] = {}
        self._lock = Lock()
        self._disabled = os.getenv("RERANKER_ENABLED", "true").lower() in {"false", "0", "no"}

    def _get_model(self) -> Any:
        """Lazy thread-safe initialization of the Cross-Encoder model."""
        if self._disabled:
            return None

        if self._model is None:
            with self._lock:
                if self._model is None:
                    try:
                        from sentence_transformers import CrossEncoder

                        logger.info(f"Loading CrossEncoder model: {self.model_name}")
                        self._model = CrossEncoder(self.model_name)
                    except Exception as err:
                        logger.warning(
                            f"Failed to load primary CrossEncoder '{self.model_name}': {err}. "
                            "Attempting lightweight fallback 'cross-encoder/ms-marco-MiniLM-L-6-v2'..."
                        )
                        try:
                            from sentence_transformers import CrossEncoder

                            self._model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
                        except Exception as inner_err:
                            logger.error(
                                f"CrossEncoder initialization failed: {inner_err}. "
                                "Falling back to fusion ranking without cross-encoder scoring."
                            )
                            self._disabled = True
                            self._model = None

        return self._model

    def _pair_hash(self, query: str, text: str) -> str:
        h = hashlib.md5()
        h.update(query.strip().lower().encode("utf-8"))
        h.update(text[:1000].encode("utf-8"))
        return h.hexdigest()

    def rerank(
        self,
        query: str,
        documents: list[Document] | list[tuple[Document, float]],
        top_k: int | None = None,
    ) -> list[tuple[Document, float, int]]:
        """Rerank candidates using Cross-Encoder in batches with score caching."""
        k = top_k if top_k is not None else self.rerank_top_k
        if not documents or not query.strip():
            return []

        doc_list: list[Document] = []
        for item in documents:
            if isinstance(item, tuple):
                doc_list.append(item[0])
            else:
                doc_list.append(item)

        if not doc_list:
            return []

        model = self._get_model()

        if model is None:
            return [(doc, 1.0 - (idx * 0.05), idx + 1) for idx, doc in enumerate(doc_list[:k])]

        try:
            # Check cached scores first to avoid redundant model inference
            scored_docs: list[tuple[Document, float]] = []
            uncached_pairs: list[list[str]] = []
            uncached_indices: list[int] = []

            with self._lock:
                for idx, doc in enumerate(doc_list):
                    phash = self._pair_hash(query, doc.page_content)
                    if phash in self._score_cache:
                        scored_docs.append((doc, self._score_cache[phash]))
                    else:
                        uncached_pairs.append([query, doc.page_content[:3000]])
                        uncached_indices.append(idx)

            # Predict uncached pairs in batch
            if uncached_pairs:
                scores = model.predict(uncached_pairs)
                with self._lock:
                    for i, idx in enumerate(uncached_indices):
                        doc = doc_list[idx]
                        score = float(scores[i]) if hasattr(scores, "__getitem__") else float(scores)
                        phash = self._pair_hash(query, doc.page_content)
                        self._score_cache[phash] = score
                        scored_docs.append((doc, score))

            # Sort descending by cross-encoder score
            scored_docs.sort(key=lambda x: x[1], reverse=True)

            results: list[tuple[Document, float, int]] = []
            for rank, (doc, score) in enumerate(scored_docs[:k], start=1):
                results.append((doc, score, rank))

            return results
        except Exception as err:
            logger.warning(f"Cross-Encoder reranking error: {err}. Preserving fusion rank order.")
            return [(doc, 1.0 - (idx * 0.05), idx + 1) for idx, doc in enumerate(doc_list[:k])]


global_reranker = CrossEncoderReranker()
