from langchain_core.documents import Document


CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200


def split_documents(documents: list[Document]) -> list[Document]:
    """Split documents without importing the heavyweight ML splitter package."""
    chunks: list[Document] = []
    step = CHUNK_SIZE - CHUNK_OVERLAP

    for document in documents:
        text = document.page_content.strip()
        if not text:
            continue

        for start in range(0, len(text), step):
            chunk_text = text[start : start + CHUNK_SIZE]
            if not chunk_text:
                continue

            metadata = dict(document.metadata)
            metadata["chunk_start"] = start
            chunks.append(Document(page_content=chunk_text, metadata=metadata))

            if start + CHUNK_SIZE >= len(text):
                break

    return chunks
