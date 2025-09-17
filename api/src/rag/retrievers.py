from typing import List, Dict, Any
import inspect
from weaviate.classes.query import Rerank
from .weav_client import get_collection
from .ingest import embed_texts

def _embed_query(query: str):
    return embed_texts([query])[0]

def _call_near_vector(collection, vec, **kwargs):
    f = collection.query.near_vector
    params = inspect.signature(f).parameters
    if "near_vector" in params:
        return f(near_vector=vec, **kwargs)
    elif "vector" in params:
        return f(vector=vec, **kwargs)
    else:
        # fallback to positional if a future client makes the vector positional-only
        return f(vec, **kwargs)

def semantic(collection, query: str, top_k: int):
    vec = _embed_query(query)
    return _call_near_vector(collection, vec, limit=top_k)

def semantic_with_rerank(collection, query: str, top_k: int, rerank_property: str):
    vec = _embed_query(query)
    rr = Rerank(query=query, prop=rerank_property)
    return _call_near_vector(collection, vec, limit=top_k, rerank=rr)

def bm25(collection, query: str, top_k: int):
    return collection.query.bm25(query=query, limit=top_k)

def hybrid(collection, query: str, top_k: int, alpha: float):
    vec = _embed_query(query)
    return collection.query.hybrid(query=query, vector=vec, limit=top_k, alpha=alpha)

def to_props(result) -> List[Dict[str, Any]]:
    return [obj.properties for obj in result.objects or []]


