"""
ingestion/ingest.py

Stream Yelp reviews JSON line-by-line, embed in batches using
nomic-embed-text-v1.5 (MRL, 256-dim) on MPS via mlx-embedding-models,
and upsert into ChromaDB. Checkpoints after every batch — safe to interrupt
and resume.

Usage:
    python -m ingestion.ingest --filepath /path/to/yelp_academic_dataset_review.json
"""
import argparse
import json
import pickle
from pathlib import Path

import chromadb
import numpy as np
from mlx_embedding_models.embedding import EmbeddingModel

from config import settings

CHECKPOINT_FILE = Path("ingestion/checkpoint.pkl")
BATCH_SIZE = 500


def load_checkpoint() -> int:
    if CHECKPOINT_FILE.exists():
        return pickle.loads(CHECKPOINT_FILE.read_bytes())
    return 0


def save_checkpoint(idx: int) -> None:
    CHECKPOINT_FILE.write_bytes(pickle.dumps(idx))


def stream_reviews(filepath: str):
    with open(filepath, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def embed(model: EmbeddingModel, texts: list[str]) -> list[list[float]]:
    embeddings = model.encode(texts, show_progress=False)
    # MRL (Matryoshka Representation Learning) truncation: slice to target dimensions, then re-
    # normalise. store vectors at 128 or 256 dimensions (instead of the standard 768) to save 
    # disk/RAM, but it still retains ~95% of its 768-dim performance.
    embeddings = embeddings[:, : settings.embed_dimensions]
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / np.where(norms == 0, 1, norms)
    return embeddings.tolist()


def ingest(filepath: str) -> None:
    client = chromadb.PersistentClient(path=settings.chroma_path)
    collection = client.get_or_create_collection(
        settings.chroma_collection,
        metadata={"hnsw:M": 16, "hnsw:construction_ef": 100},
    )
    model = EmbeddingModel.from_registry(settings.embed_model)

    start = load_checkpoint()
    batch: dict = {"ids": [], "embeddings": [], "documents": [], "metadatas": []}
    idx = 0

    for idx, review in enumerate(stream_reviews(filepath)):
        if idx < start:
            continue

        batch["ids"].append(review["review_id"])
        batch["documents"].append(review["text"])
        batch["metadatas"].append({
            "business_id": review["business_id"],
            "stars": float(review["stars"]),
            "date": review["date"],
        })

        if len(batch["ids"]) == BATCH_SIZE:
            batch["embeddings"] = embed(model, batch["documents"])
            collection.upsert(**batch)
            batch = {"ids": [], "embeddings": [], "documents": [], "metadatas": []}
            save_checkpoint(idx + 1)
            print(f"Ingested {idx + 1} reviews")

    if batch["ids"]:
        batch["embeddings"] = embed(model, batch["documents"])
        collection.upsert(**batch)
        save_checkpoint(idx + 1)
        print(f"Ingested {idx + 1} reviews (final batch)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--filepath", required=True)
    args = parser.parse_args()
    ingest(args.filepath)
