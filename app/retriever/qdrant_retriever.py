
from app.vectorstore.croma_store import get_vectorstore


def get_retriever():
    """Return a retriever without initializing embeddings during import."""
    return get_vectorstore().as_retriever(
        search_type="similarity",
        search_kwargs={"k": 5},
    )
