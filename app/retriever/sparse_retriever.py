"""Sparse keyword retriever using BM25 with strict session/thread isolation."""

from __future__ import annotations

import logging
import os
import re
from threading import Lock
from typing import Any
from langchain_core.documents import Document
from rank_bm25 import BM25Plus

logger = logging.getLogger("SparseRetriever")

# Tokenizer helper
TOKEN_PATTERN = re.compile(r"\b\w+\b")


def tokenize_text(text: str) -> list[str]:
    """Tokenize text into lowercase alphanumeric tokens for BM25."""
    return [token.lower() for token in TOKEN_PATTERN.findall(text or "")]


class SparseRetriever:
    """Session-isolated BM25 sparse keyword retriever."""

    def __init__(self, default_k: int = 30) -> None:
        self.default_k = int(os.getenv("HYBRID_TOP_K", str(default_k)))
        self._enabled = os.getenv("BM25_ENABLED", "true").lower() in {"true", "1", "yes"}
        self._documents: dict[str, list[Document]] = {}
        self._doc_tokens: dict[str, list[set[str]]] = {}
        self._bm25_indices: dict[str, BM25Plus] = {}
        self._lock = Lock()

    def index_documents(self, documents: list[Document], session_id: str) -> None:
        """Add or update documents in the session's BM25 index."""
        if not self._enabled or not documents or not session_id:
            return

        with self._lock:
            if session_id not in self._documents:
                self._documents[session_id] = []

            # Append new unique documents by page_content
            existing_contents = {d.page_content for d in self._documents[session_id]}
            for doc in documents:
                if doc.page_content not in existing_contents:
                    self._documents[session_id].append(doc)
                    existing_contents.add(doc.page_content)

            # Rebuild session BM25 index
            corpus_tokens = [tokenize_text(doc.page_content) for doc in self._documents[session_id]]
            self._doc_tokens[session_id] = [set(toks) for toks in corpus_tokens]
            if corpus_tokens:
                self._bm25_indices[session_id] = BM25Plus(corpus_tokens)

    def search(
        self,
        query: str,
        session_id: str,
        k: int | None = None,
        filter_dict: dict[str, Any] | None = None,
    ) -> list[tuple[Document, float]]:
        """Retrieve top sparse keyword matches for query within session, optionally filtered by source metadata."""
        if not self._enabled or not query or not query.strip() or not session_id:
            return []

        top_k = k if k is not None else self.default_k
        query_tokens = tokenize_text(query)
        if not query_tokens:
            return []

        query_set = set(query_tokens)

        with self._lock:
            bm25 = self._bm25_indices.get(session_id)
            docs = self._documents.get(session_id, [])
            doc_sets = self._doc_tokens.get(session_id, [])

            if not bm25 or not docs:
                # Fallback to search across all indexed sessions
                if self._documents:
                    all_docs = []
                    all_sets = []
                    for sid, sdocs in self._documents.items():
                        all_docs.extend(sdocs)
                        all_sets.extend(self._doc_tokens.get(sid, []))
                    if all_docs:
                        corpus_tokens = [tokenize_text(doc.page_content) for doc in all_docs]
                        bm25 = BM25Plus(corpus_tokens)
                        docs = all_docs
                        doc_sets = all_sets
                if not bm25 or not docs:
                    return []

            scores = bm25.get_scores(query_tokens)

        # Only retain documents that have at least one token matching the query and match filter_dict
        scored_pairs: list[tuple[Document, float]] = []
        for i in range(len(docs)):
            doc = docs[i]
            # Metadata filter check (e.g. source_id or source_type)
            if filter_dict:
                match_filter = True
                for fk, fv in filter_dict.items():
                    if doc.metadata.get(fk) != fv:
                        match_filter = False
                        break
                if not match_filter:
                    continue

            if doc_sets[i].intersection(query_set):
                scored_pairs.append((doc, float(scores[i])))

        if not scored_pairs:
            return []

        scored_pairs.sort(key=lambda x: x[1], reverse=True)

        # Normalize scores to [0, 1]
        max_score = scored_pairs[0][1] if scored_pairs else 1.0
        results: list[tuple[Document, float]] = []
        for doc, score in scored_pairs[:top_k]:
            norm_score = (score / max_score) if max_score > 0 else 0.0
            results.append((doc, norm_score))

        return results

    def clear(self, session_id: str | None = None) -> None:
        """Clear BM25 index for a session or globally."""
        with self._lock:
            if session_id:
                self._documents.pop(session_id, None)
                self._doc_tokens.pop(session_id, None)
                self._bm25_indices.pop(session_id, None)
            else:
                self._documents.clear()
                self._doc_tokens.clear()
                self._bm25_indices.clear()


global_sparse_retriever = SparseRetriever()
