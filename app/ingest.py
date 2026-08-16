from ingestion.loader import load_document
from ingestion.web_loader import load_web_page
from ingestion.youtube_loader import load_youtube
from chunking.splitter import split_documents
from vectorstore.croma_store import get_vectorstore


def ingest_document(document, session_id: str):
    """Chunk and index a document, keeping it isolated to one session/thread."""
    if not session_id or not session_id.strip():
        raise ValueError("A session_id is required to index a document.")

    # Set this before splitting so every generated chunk carries the same scope.
    document.metadata = {**document.metadata, "session_id": session_id.strip()}
    chunks = split_documents([document])
    get_vectorstore().add_documents(chunks)
    return chunks

def ingest_file(file_path, session_id: str):
    try:
    # File -> Documents
        document = load_document(file_path)

    except Exception as e:
            raise RuntimeError(
            f"Failed loading {file_path}: {e}"
        )
    return ingest_document(document, session_id)


def ingest_url(url: str, session_id: str):
    """Index a YouTube transcript or readable text from a public web page."""
    document = load_youtube(url) if ("youtube.com" in url.lower() or "youtu.be" in url.lower()) else load_web_page(url)
    return ingest_document(document, session_id)
    
