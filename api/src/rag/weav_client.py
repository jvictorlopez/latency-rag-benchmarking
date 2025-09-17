import weaviate
from weaviate.classes.config import Property, DataType, Configure
from typing import Optional
from ..settings import WEAVIATE_HTTP_HOST, WEAVIATE_HTTP_PORT, WEAVIATE_GRPC_HOST, WEAVIATE_GRPC_PORT

CLASS_NAME = "DocChunk"

def get_client():
    return weaviate.connect_to_local(
        host=WEAVIATE_HTTP_HOST,
        port=WEAVIATE_HTTP_PORT,
        grpc_port=WEAVIATE_GRPC_PORT,
    )

def ensure_schema(client: weaviate.WeaviateClient):
    try:
        listed = client.collections.list_all()
        # weaviate-client 4.7 returns a dict; newer versions return object with .collections
        if isinstance(listed, dict):
            items = listed.get("collections", [])
            existing = [it.get("name") if isinstance(it, dict) else str(it) for it in items]
        else:
            existing = [c.name for c in listed.collections]
        if CLASS_NAME in existing:
            return
    except Exception:
        # if listing fails, try create anyway
        pass

    try:
        client.collections.create(
            name=CLASS_NAME,
            vectorizer_config=Configure.Vectorizer.none(),
            properties=[
                Property(name="doc_id", data_type=DataType.TEXT),
                Property(name="source", data_type=DataType.TEXT),
                Property(name="title", data_type=DataType.TEXT),
                Property(name="page", data_type=DataType.INT),
                Property(name="chunk_index", data_type=DataType.INT),
                Property(name="chunk", data_type=DataType.TEXT),
                Property(name="created_at", data_type=DataType.DATE),
                Property(name="mime", data_type=DataType.TEXT),
                Property(name="hash", data_type=DataType.TEXT),
                Property(name="num_tokens", data_type=DataType.INT),
            ],
        )
    except Exception:
        # swallow 422 already exists
        pass

def get_collection(client):
    return client.collections.get(CLASS_NAME)


