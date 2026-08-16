from app.vectorstore.client import client
from app.vectorstore.collections import COLLECTION_NAME


count = client.count(
    collection_name=COLLECTION_NAME
)

print(count)
