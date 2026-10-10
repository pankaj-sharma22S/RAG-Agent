"""Hierarchical text chunking with atomic multimodal units."""

from __future__ import annotations

from typing import List

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


PARENT_CHUNK_SIZE = 1000
PARENT_CHUNK_OVERLAP = 100
CHILD_CHUNK_SIZE = 250
CHILD_CHUNK_OVERLAP = 30

_PARENT_SPLITTER = RecursiveCharacterTextSplitter(
    chunk_size=PARENT_CHUNK_SIZE,
    chunk_overlap=PARENT_CHUNK_OVERLAP,
)
_CHILD_SPLITTER = RecursiveCharacterTextSplitter(
    chunk_size=CHILD_CHUNK_SIZE,
    chunk_overlap=CHILD_CHUNK_OVERLAP,
)


def _document_id(document: Document, document_index: int) -> str:
    """Use a stable source identity when available, incorporating page numbers if present."""
    metadata = document.metadata or {}
    base_id = str(
        metadata.get("doc_id")
        or metadata.get("source_id")
        or metadata.get("source")
        or f"doc{document_index}"
    )
    if "page" in metadata:
        return f"{base_id}_page{metadata['page']}"
    if document_index > 0:
        return f"{base_id}_d{document_index}"
    return base_id



def _atomic_chunk(document: Document, parent_id: str) -> Document:
    """Keep images and tables intact while adding hierarchy metadata."""
    metadata = dict(document.metadata or {})
    metadata.update(
        {
            "parent_id": parent_id,
            "child_id": f"{parent_id}_c0",
            "parent_content": document.page_content,
            "chunk_level": "child",
        }
    )
    return Document(page_content=document.page_content, metadata=metadata)


def split_documents(documents: List[Document]) -> List[Document]:
    """Create child chunks with parent context, preserving atomic media units.

    Images and tables are returned as one child-sized record without splitting.
    Text documents are split first into 1,000-character parents and then into
    250-character children within each parent.
    """
    chunks: List[Document] = []

    for document_index, document in enumerate(documents):
        metadata = document.metadata or {}
        doc_id = _document_id(document, document_index)

        if metadata.get("source_type") == "image" or metadata.get("is_table") is True:
            chunks.append(_atomic_chunk(document, f"{doc_id}_p0"))
            continue

        if not document.page_content or not document.page_content.strip():
            continue

        parent_texts = _PARENT_SPLITTER.split_text(document.page_content)
        for parent_index, parent_text in enumerate(parent_texts):
            parent_id = f"{doc_id}_p{parent_index}"
            child_texts = _CHILD_SPLITTER.split_text(parent_text)

            for child_index, child_text in enumerate(child_texts):
                child_metadata = dict(metadata)
                child_metadata.update(
                    {
                        "parent_id": parent_id,
                        "child_id": f"{parent_id}_c{child_index}",
                        "parent_content": parent_text,
                        "chunk_level": "child",
                    }
                )
                chunks.append(Document(page_content=child_text, metadata=child_metadata))

    return chunks
