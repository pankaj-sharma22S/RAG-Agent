"""Weighted LangChain hybrid retrieval with parent-context resolution."""

from __future__ import annotations

import logging
import os
from typing import Any, List

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import Field

try:
    from langchain_classic.retrievers import EnsembleRetriever
except ModuleNotFoundError:
    from langchain.retrievers import EnsembleRetriever

try:
    from app.retriever.sparse_retriever import global_sparse_retriever
    from app.vectorstore.croma_store import get_vectorstore
except ModuleNotFoundError:
    from retriever.sparse_retriever import global_sparse_retriever
    from vectorstore.croma_store import get_vectorstore


logger = logging.getLogger("HybridRetriever")


class _SparseRetrieverAdapter(BaseRetriever):
    """Expose the existing session-aware BM25 retriever as a LangChain retriever."""

    session_id: str = ""
    filter_dict: dict[str, Any] = Field(default_factory=dict)
    top_k: int = 8

    def _get_relevant_documents(self, query: str, *, run_manager: Any = None) -> List[Document]:
        results = global_sparse_retriever.search(
            query=query,
            session_id=self.session_id,
            k=self.top_k,
            filter_dict=self.filter_dict or None,
        )
        return [document for document, _score in results]


def resolve_parent_documents(child_docs: List[Document]) -> List[Document]:
    """Replace ranked child chunks with one ranked macro document per parent.

    The first child encountered determines the parent rank and contributes its
    metadata, so visual fields such as ``image_url`` and ``source_type`` remain
    available to downstream renderers.
    """
    resolved: List[Document] = []
    seen_parent_ids: set[str] = set()

    for index, child in enumerate(child_docs):
        metadata = dict(child.metadata or {})
        parent_id = metadata.get("parent_id")
        parent_content = metadata.get("parent_content")

        # Documents without hierarchy metadata are still valid retrieval hits;
        # use their own content as an atomic parent and deduplicate by position.
        if parent_id is None:
            parent_id = f"__atomic_{index}"
        parent_id = str(parent_id)
        if parent_id in seen_parent_ids:
            continue
        seen_parent_ids.add(parent_id)

        if not isinstance(parent_content, str) or not parent_content:
            parent_content = child.page_content

        metadata.update(
            {
                "parent_id": parent_id,
                "parent_content": parent_content,
                "chunk_level": "parent",
            }
        )
        resolved.append(Document(page_content=parent_content, metadata=metadata))

    return resolved


def _session_filter(session_id: str, filter_dict: dict[str, Any] | None) -> dict[str, Any] | None:
    clauses: list[dict[str, Any]] = []
    if session_id:
        clauses.append({"session_id": session_id})
    if filter_dict:
        clauses.extend({key: value} for key, value in filter_dict.items() if key and value is not None)
    if not clauses:
        return None
    return clauses[0] if len(clauses) == 1 else {"$and": clauses}


class HybridRetriever:
    """Fuse dense Chroma and session-aware BM25 retrieval with weighted RRF."""

    def __init__(self) -> None:
        self.hybrid_top_k = int(os.getenv("HYBRID_TOP_K", "30"))
        self.hybrid_enabled = os.getenv("HYBRID_ENABLED", "true").lower() in {"true", "1", "yes"}

    def retrieve(
        self,
        query: str,
        session_id: str,
        top_k: int | None = None,
        filter_dict: dict[str, Any] | None = None,
    ) -> tuple[list[tuple[Document, float]], int, int]:
        """Retrieve, fuse, resolve parents, and return ranked context documents."""
        if not query or not query.strip():
            return [], 0, 0

        result_limit = top_k if top_k is not None else self.hybrid_top_k
        try:
            store = get_vectorstore()
            dense_kwargs: dict[str, Any] = {"k": 8}
            metadata_filter = _session_filter(session_id, filter_dict)
            if metadata_filter:
                dense_kwargs["filter"] = metadata_filter
            dense_retriever = store.as_retriever(search_kwargs=dense_kwargs)

            sparse_retriever = _SparseRetrieverAdapter(
                session_id=session_id,
                filter_dict=filter_dict or {},
                top_k=8,
            )
            dense_docs = dense_retriever.invoke(query)
            dense_count = len(dense_docs)

            if self.hybrid_enabled:
                sparse_docs = sparse_retriever.invoke(query)
                sparse_count = len(sparse_docs)
                ensemble = EnsembleRetriever(
                    retrievers=[dense_retriever, sparse_retriever],
                    weights=[0.6, 0.4],
                )
                child_docs = ensemble.invoke(query)
            else:
                sparse_count = 0
                child_docs = dense_docs

            parent_docs = resolve_parent_documents(child_docs[:result_limit])
            ranked = [
                (document, 1.0 / (rank + 1))
                for rank, document in enumerate(parent_docs)
            ]
            return ranked, dense_count, sparse_count
        except Exception as error:
            logger.warning("Hybrid retrieval failed: %s", error)
            return [], 0, 0


global_hybrid_retriever = HybridRetriever()
