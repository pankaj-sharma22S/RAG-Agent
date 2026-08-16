"""Hybrid retriever combining Dense (BGE-M3 + Chroma) and Sparse (BM25) in parallel with Reciprocal Rank Fusion."""

from __future__ import annotations

import logging
import os
from concurrent.futures import ThreadPoolExecutor
from typing import Any
from langchain_core.documents import Document

try:
    from app.retriever.dense_retriever import global_dense_retriever
    from app.retriever.sparse_retriever import global_sparse_retriever
except ModuleNotFoundError:
    from retriever.dense_retriever import global_dense_retriever
    from retriever.sparse_retriever import global_sparse_retriever

logger = logging.getLogger("HybridRetriever")


class HybridRetriever:
    """Executes dense and sparse searches in parallel and fuses them using Reciprocal Rank Fusion (RRF)."""

    def __init__(self) -> None:
        self.rrf_k = int(os.getenv("RRF_K", "60"))
        self.hybrid_top_k = int(os.getenv("HYBRID_TOP_K", "30"))
        self.hybrid_enabled = os.getenv("HYBRID_ENABLED", "true").lower() in {"true", "1", "yes"}

    def retrieve(
        self,
        query: str,
        session_id: str,
        top_k: int | None = None,
        filter_dict: dict[str, Any] | None = None,
    ) -> tuple[list[tuple[Document, float]], int, int]:
        """Perform parallel hybrid retrieval and return fused documents along with counts."""
        k = top_k if top_k is not None else self.hybrid_top_k

        if not self.hybrid_enabled:
            dense_results = global_dense_retriever.search(
                query=query,
                session_id=session_id,
                k=k,
                filter_dict=filter_dict,
            )
            return dense_results, len(dense_results), 0

        # Execute dense and sparse retrieval in parallel
        with ThreadPoolExecutor(max_workers=2) as executor:
            dense_future = executor.submit(
                global_dense_retriever.search,
                query=query,
                session_id=session_id,
                k=k,
                filter_dict=filter_dict,
            )
            sparse_future = executor.submit(
                global_sparse_retriever.search,
                query=query,
                session_id=session_id,
                k=k,
                filter_dict=filter_dict,
            )
            dense_results = dense_future.result()
            sparse_results = sparse_future.result()

        dense_count = len(dense_results)
        sparse_count = len(sparse_results)

        # Reciprocal Rank Fusion (RRF)
        fused = self.reciprocal_rank_fusion(
            dense_results=dense_results,
            sparse_results=sparse_results,
            rrf_k=self.rrf_k,
            top_k=k,
        )

        return fused, dense_count, sparse_count

    def reciprocal_rank_fusion(
        self,
        dense_results: list[tuple[Document, float]],
        sparse_results: list[tuple[Document, float]],
        rrf_k: int = 60,
        top_k: int = 30,
    ) -> list[tuple[Document, float]]:
        """Apply Reciprocal Rank Fusion to rank-merge multiple retrieval result lists."""
        doc_scores: dict[str, float] = {}
        doc_lookup: dict[str, Document] = {}

        # Process dense ranking (1-based rank)
        for rank, (doc, _dense_score) in enumerate(dense_results, start=1):
            key = doc.page_content.strip()
            doc_lookup[key] = doc
            doc_scores[key] = doc_scores.get(key, 0.0) + (1.0 / (rrf_k + rank))

        # Process sparse ranking (1-based rank)
        for rank, (doc, _sparse_score) in enumerate(sparse_results, start=1):
            key = doc.page_content.strip()
            if key not in doc_lookup:
                doc_lookup[key] = doc
            doc_scores[key] = doc_scores.get(key, 0.0) + (1.0 / (rrf_k + rank))

        # Sort by combined RRF score descending
        sorted_keys = sorted(doc_scores.keys(), key=lambda k: doc_scores[k], reverse=True)

        fused_list: list[tuple[Document, float]] = []
        for k_idx in sorted_keys[:top_k]:
            fused_list.append((doc_lookup[k_idx], doc_scores[k_idx]))

        return fused_list


global_hybrid_retriever = HybridRetriever()
