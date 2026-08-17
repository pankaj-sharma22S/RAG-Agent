"""Chroma vector store initialization with resilient import fallback."""

from __future__ import annotations

from pathlib import Path
from threading import Lock

try:
    from langchain_chroma import Chroma
except ModuleNotFoundError:
    try:
        from langchain_community.vectorstores import Chroma
    except ModuleNotFoundError as err:
        raise ModuleNotFoundError(
            "Neither 'langchain-chroma' nor 'langchain-community' is installed. "
            "Please install langchain-chroma using: pip install langchain-chroma"
        ) from err

# The API starts from ``app`` while the workflow starts from the repository
# root. Resolve the embedding module correctly in either launch mode.
try:
    from app.embedding.embedding_model import get_embedding_model
except ModuleNotFoundError:
    from embedding.embedding_model import get_embedding_model


PERSIST_DIRECTORY = Path(__file__).resolve().parent.parent / "data" / "chroma"

_vectorstore = None
_vectorstore_lock = Lock()
_chroma_init_count = 0
_chroma_first_init_ms = 0.0


def get_vectorstore():
    """Create the persistent Chroma store only when it is first needed."""
    global _vectorstore, _chroma_init_count, _chroma_first_init_ms

    if _vectorstore is None:
        with _vectorstore_lock:
            if _vectorstore is None:
                import time
                t0 = time.perf_counter()
                _vectorstore = Chroma(
                    collection_name="rag_documents_v2",
                    embedding_function=get_embedding_model(),
                    persist_directory=str(PERSIST_DIRECTORY),
                )
                _chroma_first_init_ms = round((time.perf_counter() - t0) * 1000, 2)
                _chroma_init_count += 1

    return _vectorstore


def get_chroma_lifecycle_stats() -> dict[str, Any]:
    """Return lifecycle metrics for the Chroma persistent store."""
    count = _vectorstore._collection.count() if _vectorstore and hasattr(_vectorstore, "_collection") else 0
    return {
        "component": "ChromaStore",
        "initialization_count": _chroma_init_count,
        "first_init_ms": _chroma_first_init_ms,
        "collection_items": count,
    }
