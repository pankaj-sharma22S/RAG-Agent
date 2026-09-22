"""Persistent local Chroma vector store backed by FastEmbed embeddings."""

from __future__ import annotations

import time
import logging
import os
from pathlib import Path
from threading import Lock
from typing import Any

from langchain_core.vectorstores import VectorStore

try:
    from langchain_chroma import Chroma
except ModuleNotFoundError:
    try:
        from langchain_community.vectorstores import Chroma
    except ModuleNotFoundError as error:
        raise ModuleNotFoundError(
            "A Chroma integration is required. Install 'langchain-chroma' or "
            "'langchain-community'."
        ) from error


PERSIST_DIRECTORY = Path("./local_vector_db")
COLLECTION_NAME = "enterprise_knowledge"
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"

_vectorstore: VectorStore | None = None
_vectorstore_lock = Lock()
_chroma_init_count = 0
_chroma_first_init_ms = 0.0
logger = logging.getLogger(__name__)


def _create_vectorstore() -> VectorStore:
    """Construct the local Chroma store with an ONNX FastEmbed model."""
    PERSIST_DIRECTORY.mkdir(parents=True, exist_ok=True)
    embeddings = None

    # FastEmbed downloads its ONNX model on first use. Avoid a long retry
    # cycle in offline deployments when no FastEmbed cache exists.
    fastembed_cache = Path(
        os.environ.get(
            "FASTEMBED_CACHE_PATH", Path.home() / ".cache" / "fastembed"
        )
    )
    if fastembed_cache.exists():
        try:
            from langchain_community.embeddings.fastembed import FastEmbedEmbeddings

            embeddings = FastEmbedEmbeddings(model_name=EMBEDDING_MODEL)
        except Exception as error:
            logger.warning("FastEmbed initialization failed; using local embeddings: %s", error)

    if embeddings is None:
        try:
            from app.embedding.embedding_model import get_embedding_model
        except ModuleNotFoundError:
            from embedding.embedding_model import get_embedding_model
        embeddings = get_embedding_model()

    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=str(PERSIST_DIRECTORY),
    )


def get_vectorstore() -> VectorStore:
    """Return the singleton persistent local vector store."""
    global _vectorstore, _chroma_init_count, _chroma_first_init_ms

    if _vectorstore is None:
        with _vectorstore_lock:
            if _vectorstore is None:
                started_at = time.perf_counter()
                _vectorstore = _create_vectorstore()
                _chroma_first_init_ms = round((time.perf_counter() - started_at) * 1000, 2)
                _chroma_init_count += 1

    return _vectorstore


def get_chroma_lifecycle_stats() -> dict[str, Any]:
    """Return lifecycle metrics for the persistent Chroma collection."""
    store = _vectorstore
    collection_items = 0
    if store is not None and hasattr(store, "_collection"):
        collection_items = store._collection.count()

    return {
        "component": "ChromaStore",
        "initialization_count": _chroma_init_count,
        "first_init_ms": _chroma_first_init_ms,
        "collection_items": collection_items,
        "persist_directory": str(PERSIST_DIRECTORY),
        "collection_name": COLLECTION_NAME,
    }
