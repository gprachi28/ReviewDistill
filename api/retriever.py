"""
api/retriever.py

Semantic retrieval: embed a query and find the most relevant review snippets
within a candidate pool of businesses.

Public API:
    retrieve(semantic_query, business_ids=None, top_k=20) -> list[dict]
"""
import chromadb
from mlx_embedding_models.embedding import EmbeddingModel

from config import settings
from ingestion.ingest_nola import embed

QUERY_PREFIX = "search_query: "

# Lazy singletons — loaded once on first call, reused for every request.
_model: EmbeddingModel | None = None
_collection = None


def _get_model() -> EmbeddingModel:
    global _model
    if _model is None:
        _model = EmbeddingModel.from_registry(settings.embed_model)
    return _model


def _get_collection():
    global _collection
    if _collection is None:
        client = chromadb.PersistentClient(path=settings.chroma_path)
        _collection = client.get_collection(settings.chroma_collection)
    return _collection


def retrieve(
    semantic_query: str,
    business_ids: list[str] | None = None,
    top_k: int = 20,
) -> list[dict]:
    """
    Return the top-K review snippets most similar to semantic_query.

    Args:
        semantic_query: Free-text phrase to embed and search (from query planner).
        business_ids:   Candidate pool from SQL filter. If None, searches all businesses.
        top_k:          Maximum number of snippets to return.

    Returns:
        List of dicts with keys: business_id, text, stars, distance.
        Sorted by distance ascending (closest match first).
    """
    model = _get_model()
    collection = _get_collection()

    query_text = QUERY_PREFIX + semantic_query
    query_embedding = embed(model, [query_text])[0]

    where = {"business_id": {"$in": business_ids}} if business_ids else None

    # Cap n_results to the collection size to avoid ChromaDB errors on small pools.
    n = min(top_k, collection.count())
    if n == 0:
        return []

    kwargs = dict(query_embeddings=[query_embedding], n_results=n, include=["documents", "metadatas", "distances"])
    if where:
        kwargs["where"] = where

    results = collection.query(**kwargs)

    snippets = []
    ids_list = results["ids"][0]
    docs = results["documents"][0]
    metas = results["metadatas"][0]
    dists = results["distances"][0]

    for doc, meta, dist in zip(docs, metas, dists):
        snippets.append({
            "business_id": meta["business_id"],
            "text": doc,
            "stars": meta["stars"],
            "distance": dist,
        })

    return snippets
