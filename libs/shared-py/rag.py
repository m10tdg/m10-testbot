# libs/shared-py/rag.py
def retrieve_context(qdrant: QdrantClient, tenant_id: str, project_id: str,
                      query_vector: list[float], types: list[str] | None = None, top_k: int = 5):
    collection = f"{tenant_id}_{project_id}"
    query_filter = None
    if types:
        from qdrant_client.models import Filter, FieldCondition, MatchAny
        query_filter = Filter(must=[FieldCondition(key="type", match=MatchAny(any=types))])
    hits = qdrant.search(
        collection_name=collection,   # <-- this is the whole isolation guarantee, right here
        query_vector=query_vector,
        query_filter=query_filter,
        limit=top_k,
    )
    return [h.payload["text"] for h in hits]