from qdrant_client.models import (
    Distance,
    VectorParams
)

from app.vectorstore.client import client


COLLECTION_NAME = "rag_documents"


def create_collection():

    collections = client.get_collections()

    names = [
        collection.name
        for collection in collections.collections
    ]

    if COLLECTION_NAME in names:
        return

    client.create_collection(

        collection_name=COLLECTION_NAME,

        vectors_config=VectorParams(

            size=1024,

            distance=Distance.COSINE

        )

    )
