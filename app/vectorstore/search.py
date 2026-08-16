from app.vectorstore.client import client
from app.vectorstore.collections import COLLECTION_NAME


result = client.scroll(

    collection_name=COLLECTION_NAME,

    limit=5

)


for item in result[0]:

    print(item.payload)
