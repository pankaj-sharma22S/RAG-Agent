"""Document, Media, and URL Ingestion Pipeline with Lazy Indexing & Source Hash Tracking."""

from __future__ import annotations

import hashlib
import logging
from typing import Any
from chunking.splitter import split_documents
from core.document import create_document
from ingestion.loader import load_document
from ingestion.web_loader import load_web_page
from ingestion.youtube_loader import load_youtube
from vectorstore.croma_store import get_vectorstore

try:
    from app.db.service import get_active_source, set_active_source
    from app.retriever.sparse_retriever import global_sparse_retriever
    from app.tracing import safe_traceable
except ModuleNotFoundError:
    from db.service import get_active_source, set_active_source
    from retriever.sparse_retriever import global_sparse_retriever
    from tracing import safe_traceable

logger = logging.getLogger("IngestPipeline")

# Global set of indexed source hashes: (session_id, source_hash)
INDEXED_SOURCE_HASHES: set[tuple[str, str]] = set()


def sanitize_chroma_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    """Ensure metadata contains only non-null primitives or non-empty lists compatible with ChromaDB."""
    cleaned: dict[str, Any] = {}
    for key, value in metadata.items():
        if value is None:
            continue
        if isinstance(value, (str, int, float, bool)):
            cleaned[key] = value
        elif isinstance(value, list):
            if len(value) > 0 and all(isinstance(v, (str, int, float, bool)) for v in value):
                cleaned[key] = value
            elif len(value) > 0:
                cleaned[key] = [str(v) for v in value]
            # Empty lists are omitted so Chroma does not throw ValueError
        elif isinstance(value, dict):
            import json
            cleaned[key] = json.dumps(value)
        else:
            cleaned[key] = str(value)
    return cleaned


def calculate_source_hash(source_id: str, content: str) -> str:
    """Compute SHA-256 fingerprint for document content to prevent repeated re-indexing."""
    raw = f"{source_id}:{content[:50000]}".encode("utf-8", errors="ignore")
    return hashlib.sha256(raw).hexdigest()


def is_source_already_indexed(session_id: str, source_hash: str) -> bool:
    """Check whether a specific document version is already indexed in Chroma & BM25."""
    return (session_id.strip(), source_hash) in INDEXED_SOURCE_HASHES


@safe_traceable(name="Register Active Source", run_type="chain", tags=["rag-agent", "ingestion", "metadata"])
def register_active_source(document, session_id: str) -> dict[str, Any]:
    """Register source metadata and extracted content in database without running expensive vector embeddings."""
    clean_session_id = session_id.strip()
    source_url = document.metadata.get("source_url", document.metadata.get("source", ""))
    source_id = document.metadata.get("source_id", document.metadata.get("video_id", document.metadata.get("filename", source_url)))
    source_type = document.metadata.get("source_type", document.metadata.get("type", "document"))
    title = document.metadata.get("title", source_id or "Document")
    timestamp = document.metadata.get("timestamp")

    source_hash = calculate_source_hash(str(source_id), document.page_content)
    is_indexed = is_source_already_indexed(clean_session_id, source_hash)

    source_meta = {
        "source_id": str(source_id),
        "source_type": str(source_type),
        "source_url": str(source_url),
        "title": str(title),
        "source_hash": source_hash,
        "is_indexed": is_indexed,
        "extracted_text": document.page_content[:30000],  # Keep representative text for direct summary
        "session_id": clean_session_id,
        "thread_id": clean_session_id,
    }
    if timestamp is not None:
        source_meta["timestamp"] = timestamp

    try:
        set_active_source(clean_session_id, source_meta)
    except Exception as err:
        logger.warning(f"Could not persist active source to database: {err}")

    return source_meta


@safe_traceable(name="Ingest Document Pipeline", run_type="chain", tags=["rag-agent", "ingestion", "indexing"])
def ingest_document(document, session_id: str, force_reindex: bool = False):
    """Chunk, embed, and index a document into ChromaDB and BM25, refreshing index if forced."""
    if not session_id or not session_id.strip():
        raise ValueError("A session_id is required to index a document.")

    clean_session_id = session_id.strip()
    source_url = document.metadata.get("source_url", document.metadata.get("source", ""))
    source_id = document.metadata.get("source_id", document.metadata.get("video_id", document.metadata.get("filename", source_url)))
    source_type = document.metadata.get("source_type", document.metadata.get("type", "document"))
    title = document.metadata.get("title", source_id or "Document")
    timestamp = document.metadata.get("timestamp")

    source_hash = calculate_source_hash(str(source_id), document.page_content)
    already_indexed = is_source_already_indexed(clean_session_id, source_hash)

    source_meta = {
        "source_id": str(source_id),
        "source_type": str(source_type),
        "source_url": str(source_url),
        "title": str(title),
        "source_hash": source_hash,
        "is_indexed": True,
        "session_id": clean_session_id,
        "thread_id": clean_session_id,
    }
    if timestamp is not None:
        source_meta["timestamp"] = timestamp

    # Set metadata on document before splitting
    document.metadata = {**document.metadata, **source_meta}
    chunks = split_documents([document])

    # If already indexed and not forcing reindex, return existing chunk representations
    if already_indexed and not force_reindex:
        logger.info(f"Source '{source_id}' (hash: {source_hash[:8]}) already indexed for session '{clean_session_id}'. Returning {len(chunks)} chunks.")
        return chunks

    # Sanitize metadata for ChromaDB compatibility (omit empty lists, serialize dicts, drop None)
    for chunk in chunks:
        if chunk.metadata:
            chunk.metadata = sanitize_chroma_metadata(chunk.metadata)

    # Add to ChromaDB vector store
    if chunks:
        get_vectorstore().add_documents(chunks)
        # Synchronize BM25 sparse index
        global_sparse_retriever.index_documents(chunks, session_id=clean_session_id)

    # Record hash in memory to prevent duplicate indexing
    INDEXED_SOURCE_HASHES.add((clean_session_id, source_hash))

    # Persist active source for this session
    try:
        set_active_source(clean_session_id, source_meta)
    except Exception as err:
        logger.warning(f"Could not persist active source to database: {err}")

    return chunks


@safe_traceable(name="Ingest File", run_type="chain", tags=["rag-agent", "ingestion", "file"])
def ingest_file(file_path: str, session_id: str, lazy: bool = False, force_reindex: bool = True):
    """Load, scan, and index an uploaded document or media file."""
    try:
        documents = load_document(file_path)
    except Exception as e:
        raise RuntimeError(f"Failed loading {file_path}: {e}") from e

    if not documents:
        raise RuntimeError(f"No documents were produced while loading {file_path}")

    # LangChain loaders may return one document per page/sheet/record. Keep
    # those atomic units intact while preserving the existing single-document
    # ingestion API used by the rest of the application.
    for document in documents:
        register_active_source(document, session_id)

    if lazy:
        return documents

    chunks = []
    for document in documents:
        chunks.extend(ingest_document(document, session_id, force_reindex=force_reindex))
    return chunks


@safe_traceable(name="Ingest URL", run_type="chain", tags=["rag-agent", "ingestion", "url"])
def ingest_url(url: str, session_id: str, lazy: bool = False):
    """Download, scan, and extract transcript or web page, registering active source metadata.
    
    If lazy=True, extracts content and registers source without computing vector embeddings.
    If lazy=False, immediately builds ChromaDB and BM25 indices.
    """
    is_yt = "youtube.com" in url.lower() or "youtu.be" in url.lower()
    document = load_youtube(url) if is_yt else load_web_page(url)
    
    register_active_source(document, session_id)
    if not lazy:
        chunks = ingest_document(document, session_id)
        return chunks if chunks else [document]
    return [document]


def ensure_source_indexed(session_id: str) -> bool:
    """Ensure the active source for a session is indexed before executing hybrid retrieval."""
    clean_session_id = session_id.strip()
    active_src = get_active_source(clean_session_id)
    if not active_src:
        return False

    source_hash = active_src.get("source_hash", "")
    if is_source_already_indexed(clean_session_id, source_hash):
        return True

    # Build document representation and index
    extracted_text = active_src.get("extracted_text", "")
    if not extracted_text:
        return False

    doc = create_document(
        extracted_text,
        {
            "source": active_src.get("source_url", active_src.get("source_id", "document")),
            "source_id": active_src.get("source_id", "document"),
            "source_type": active_src.get("source_type", "document"),
            "title": active_src.get("title", "Document"),
            "source_url": active_src.get("source_url", ""),
            "timestamp": active_src.get("timestamp"),
        },
    )
    ingest_document(doc, session_id=clean_session_id)
    return True
