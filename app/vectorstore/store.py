from qdrant_client.models import PointStruct

from app.vectorstore.client import client
from app.vectorstore.collections import COLLECTION_NAME



def store_documents(
        documents,
        embeddings
):

    points = []


    for index, (doc, vector) in enumerate(
        zip(documents, embeddings)
    ):


        points.append(

            PointStruct(

                id=index,

                vector=vector.tolist(),

                payload={

                    "text": doc.page_content,

                    **doc.metadata

                }

            )

        )


    client.upsert(

        collection_name=COLLECTION_NAME,

        points=points

    )

