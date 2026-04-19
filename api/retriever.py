"""
api/retriever.py

Shared ChromaDB retrieval logic used by all pipeline versions.
"""
import chromadb

from config import settings

_collection = None


def get_collection():
    global _collection
    if _collection is None:
        client = chromadb.PersistentClient(path=settings.chroma_path)
        try:
            _collection = client.get_collection(settings.chroma_collection)
        except Exception as exc:
            raise RuntimeError(
                f"ChromaDB collection '{settings.chroma_collection}' not found — "
                "run ingestion first."
            ) from exc
    return _collection


def retrieve_reviews(business_id: str, n_results: int = 50) -> list[dict]:
    collection = get_collection()
    results = collection.get(
        where={"business_id": business_id},
        limit=n_results,
        include=["documents", "metadatas"],
    )
    return [
        {"text": doc, "stars": meta["stars"]}
        for doc, meta in zip(results["documents"], results["metadatas"])
    ]
