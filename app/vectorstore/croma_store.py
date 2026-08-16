from langchain_chroma import Chroma
from pathlib import Path
from threading import Lock

# The API starts from ``app`` while the workflow starts from the repository
# root.  Resolve the embedding module correctly in either launch mode.
try:
    from app.embedding.embedding_model import get_embedding_model
except ModuleNotFoundError:
    from embedding.embedding_model import get_embedding_model


PERSIST_DIRECTORY = Path(__file__).resolve().parent.parent / "data" / "chroma"

_vectorstore = None
_vectorstore_lock = Lock()


def get_vectorstore():
    """Create the persistent Chroma store only when it is first needed."""
    global _vectorstore

    if _vectorstore is None:
        with _vectorstore_lock:
            if _vectorstore is None:
                _vectorstore = Chroma(
                    collection_name="rag_documents",
                    embedding_function=get_embedding_model(),
                    persist_directory=str(PERSIST_DIRECTORY),
                )

    return _vectorstore
