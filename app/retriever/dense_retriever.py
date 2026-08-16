"""Dense vector retriever using BGE-M3 embeddings and ChromaDB persistent storage."""

from __future__ import annotations

import logging
import os
from typing import Any
from langchain_core.documents import Document

try:
    from app.vectorstore.croma_store import get_vectorstore
except ModuleNotFoundError:
    from vectorstore.croma_store import get_vectorstore

logger = logging.getLogger("DenseRetriever")


class DenseRetriever:
    """Performs semantic similarity search over ChromaDB with session isolation."""

    def __init__(self, default_k: int = 30) -> None:
        self.default_k = int(os.getenv("HYBRID_TOP_K", str(default_k)))

    def search(
        self,
        query: str,
        session_id: str,
        k: int | None = None,
        filter_dict: dict[str, Any] | None = None,
    ) -> list[tuple[Document, float]]:
        """Retrieve top dense documents with similarity relevance scores."""
        if not query or not query.strip():
            return []

        top_k = k if k is not None else self.default_k
        store = get_vectorstore()

        meta_filter: dict[str, Any] | None = None
        filter_clauses: list[dict[str, Any]] = []
        if session_id:
            filter_clauses.append({"session_id": session_id})
        if filter_dict:
            for fk, fv in filter_dict.items():
                if fk and fv is not None:
                    filter_clauses.append({fk: fv})

        if len(filter_clauses) == 1:
            meta_filter = filter_clauses[0]
        elif len(filter_clauses) > 1:
            meta_filter = {"$and": filter_clauses}

        try:
            results = store.similarity_search_with_relevance_scores(
                query,
                k=top_k,
                filter=meta_filter if meta_filter else None,
            )
            return [(doc, float(score)) for doc, score in results]
        except Exception as err:
            logger.warning(f"Dense vector retrieval encountered an error: {err}")
            # Fallback to similarity_search if score calculation fails
            try:
                raw_docs = store.similarity_search(query, k=top_k, filter=meta_filter if meta_filter else None)
                return [(doc, 0.5) for doc in raw_docs]
            except Exception as inner_err:
                logger.error(f"Fallback dense retrieval failed: {inner_err}")
                return []


global_dense_retriever = DenseRetriever()
