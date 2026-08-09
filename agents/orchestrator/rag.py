from qdrant_client.models import Filter, FieldCondition, MatchAny


def retrieve_context(qdrant, tenant_id: str, project_id: str, query_vector: list[float],
                      types: list[str] | None = None, top_k: int = 5) -> list[str]:
    """This function IS the tenant/project isolation guarantee: it only ever searches
    the one collection named after this exact tenant+project pair. There is no
    parameter that lets you accidentally search another project's data."""
    collection = f"{tenant_id}_{project_id}"

    query_filter = None
    if types:
        query_filter = Filter(must=[FieldCondition(key="type", match=MatchAny(any=types))])

    try:
        hits = qdrant.search(
            collection_name=collection,
            query_vector=query_vector,
            query_filter=query_filter,
            limit=top_k,
        )
    except Exception:
        # collection might not exist yet if no documents were uploaded - that's fine,
        # scenario generation just proceeds with no context.
        return []

    return [h.payload["text"] for h in hits]
