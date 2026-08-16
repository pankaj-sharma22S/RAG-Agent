
from langchain_huggingface import HuggingFaceEmbeddings

embedding_model = None


def get_embedding_model():

    global embedding_model

    if embedding_model is None:
        print("Loading embedding model...")

        embedding_model = HuggingFaceEmbeddings(
            model_name="BAAI/bge-m3",
            # The model is provisioned locally. Avoid a Hugging Face metadata
            # request on every API worker startup/first ingestion request.
            model_kwargs={"local_files_only": True},
        )

        print("Embedding model loaded")

    return embedding_model
