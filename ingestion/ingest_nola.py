"""
ingestion/ingest_nola.py

Ingest New Orleans restaurants from the Yelp Open Dataset into:
  - SQLite (yelp_reviews.db): structured business attributes for SQL filtering
  - ChromaDB: embedded review snippets for semantic retrieval

Usage:
    python -m ingestion.ingest_nola \
        --business /path/to/yelp_academic_dataset_business.json \
        --review   /path/to/yelp_academic_dataset_review.json
"""
import argparse
import ast
import json
import pickle
import re
import sqlite3
from pathlib import Path

import chromadb
import numpy as np
from mlx_embedding_models.embedding import EmbeddingModel

from config import settings

CHECKPOINT_FILE = Path("ingestion/checkpoint_nola.pkl")
BATCH_SIZE = 500
EMBED_PREFIX = "search_document: "


# ── Shared helpers ─────────────────────────────────────────────────────────────


def stream_reviews(filepath: str):
    with open(filepath, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def embed(model: EmbeddingModel, texts: list[str]) -> list[list[float]]:
    # Truncate to 500 tokens (mlx_embedding_models hard-caps at 512; 500 leaves room for
    # special tokens [CLS]/[SEP] added internally).
    truncated = []
    for text in texts:
        ids = model.tokenizer.encode(text, max_length=500, truncation=True)
        truncated.append(model.tokenizer.decode(ids, skip_special_tokens=True))
    embeddings = model.encode(truncated, show_progress=False)
    embeddings = embeddings[:, : settings.embed_dimensions]
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / np.where(norms == 0, 1, norms)
    return embeddings.tolist()


# ── Attribute parsing ──────────────────────────────────────────────────────────


def _parse_bool(val: str | None) -> int | None:
    """'True'/'False' string → 1/0/None."""
    if val == "True":
        return 1
    if val == "False":
        return 0
    return None


def _parse_str(val: str | None) -> str | None:
    """Strip Python repr string wrapper: u'loud' or 'loud' → 'loud'."""
    if val is None:
        return None
    m = re.match(r"^u?'(.*)'$", val.strip())
    return m.group(1) if m else val


def _parse_dict(val: str | None) -> str | None:
    """Python-repr dict string → JSON string, or None on failure."""
    if val is None:
        return None
    try:
        return json.dumps(ast.literal_eval(val))
    except (ValueError, SyntaxError):
        return None


# ── Business parsing ───────────────────────────────────────────────────────────


def parse_business(record: dict) -> dict | None:
    """
    Extract and normalize a business record for the NOLA subset.
    Returns None if the record doesn't meet filter criteria.
    Filter: city == 'New Orleans', categories contains 'Restaurant', review_count > 50.
    """
    if record.get("city") != "New Orleans":
        return None
    categories = record.get("categories") or ""
    if "Restaurant" not in categories:
        return None
    if record.get("review_count", 0) <= 50:
        return None

    attrs = record.get("attributes") or {}

    price_raw = attrs.get("RestaurantsPriceRange2")
    try:
        price_range = int(price_raw) if price_raw is not None else None
    except (ValueError, TypeError):
        price_range = None

    return {
        "business_id": record["business_id"],
        "name": record["name"],
        "stars": float(record["stars"]),
        "review_count": int(record["review_count"]),
        "categories": categories,
        "latitude": record.get("latitude"),
        "longitude": record.get("longitude"),
        "price_range": price_range,
        "noise_level": _parse_str(attrs.get("NoiseLevel")),
        "alcohol": _parse_str(attrs.get("Alcohol")),
        "attire": _parse_str(attrs.get("RestaurantsAttire")),
        "wifi": _parse_str(attrs.get("WiFi")),
        "smoking": _parse_str(attrs.get("Smoking")),
        "good_for_groups": _parse_bool(attrs.get("RestaurantsGoodForGroups")),
        "takes_reservations": _parse_bool(attrs.get("RestaurantsReservations")),
        "outdoor_seating": _parse_bool(attrs.get("OutdoorSeating")),
        "good_for_kids": _parse_bool(attrs.get("GoodForKids")),
        "good_for_dancing": _parse_bool(attrs.get("GoodForDancing")),
        "happy_hour": _parse_bool(attrs.get("HappyHour")),
        "has_tv": _parse_bool(attrs.get("HasTV")),
        "caters": _parse_bool(attrs.get("Caters")),
        "wheelchair_accessible": _parse_bool(attrs.get("WheelchairAccessible")),
        "dogs_allowed": _parse_bool(attrs.get("DogsAllowed")),
        "byob": _parse_bool(attrs.get("BYOB")),
        "corkage": _parse_bool(attrs.get("Corkage")),
        "ambience": _parse_dict(attrs.get("Ambience")),
        "good_for_meal": _parse_dict(attrs.get("GoodForMeal")),
        "music": _parse_dict(attrs.get("Music")),
        "parking": _parse_dict(attrs.get("BusinessParking")),
    }


# ── SQLite setup ───────────────────────────────────────────────────────────────


def create_schema(conn: sqlite3.Connection) -> None:
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS businesses (
            business_id          TEXT PRIMARY KEY,
            name                 TEXT NOT NULL,
            stars                REAL NOT NULL,
            review_count         INTEGER NOT NULL,
            price_range          INTEGER,
            noise_level          TEXT,
            alcohol              TEXT,
            attire               TEXT,
            wifi                 TEXT,
            smoking              TEXT,
            good_for_groups      INTEGER,
            takes_reservations   INTEGER,
            outdoor_seating      INTEGER,
            good_for_kids        INTEGER,
            good_for_dancing     INTEGER,
            happy_hour           INTEGER,
            has_tv               INTEGER,
            caters               INTEGER,
            wheelchair_accessible INTEGER,
            dogs_allowed         INTEGER,
            byob                 INTEGER,
            corkage              INTEGER,
            ambience             TEXT,
            good_for_meal        TEXT,
            music                TEXT,
            parking              TEXT,
            categories           TEXT NOT NULL,
            latitude             REAL,
            longitude            REAL
        );
        CREATE INDEX IF NOT EXISTS idx_biz_noise    ON businesses(noise_level);
        CREATE INDEX IF NOT EXISTS idx_biz_groups   ON businesses(good_for_groups);
        CREATE INDEX IF NOT EXISTS idx_biz_price    ON businesses(price_range);
        CREATE INDEX IF NOT EXISTS idx_biz_stars    ON businesses(stars);
        CREATE INDEX IF NOT EXISTS idx_biz_alcohol  ON businesses(alcohol);
        CREATE INDEX IF NOT EXISTS idx_biz_outdoor  ON businesses(outdoor_seating);
        CREATE INDEX IF NOT EXISTS idx_biz_kids     ON businesses(good_for_kids);
        CREATE INDEX IF NOT EXISTS idx_biz_attire   ON businesses(attire);
    """)
    conn.commit()


# ── Ingest steps ───────────────────────────────────────────────────────────────


def ingest_businesses(business_filepath: str) -> set[str]:
    """Parse business.json, insert NOLA restaurants into SQLite. Returns set of business_ids."""
    conn = sqlite3.connect(settings.sqlite_path)
    create_schema(conn)

    # Skip if already populated
    existing = conn.execute("SELECT COUNT(*) FROM businesses").fetchone()[0]
    if existing > 0:
        print(f"businesses table already has {existing} rows — skipping business ingest")
        ids = {row[0] for row in conn.execute("SELECT business_id FROM businesses")}
        conn.close()
        return ids

    rows = []
    for record in stream_reviews(business_filepath):
        biz = parse_business(record)
        if biz is None:
            continue
        rows.append(biz)

    conn.executemany(
        """INSERT OR IGNORE INTO businesses VALUES (
            :business_id, :name, :stars, :review_count,
            :price_range, :noise_level, :alcohol, :attire, :wifi, :smoking,
            :good_for_groups, :takes_reservations, :outdoor_seating, :good_for_kids,
            :good_for_dancing, :happy_hour, :has_tv, :caters, :wheelchair_accessible,
            :dogs_allowed, :byob, :corkage,
            :ambience, :good_for_meal, :music, :parking,
            :categories, :latitude, :longitude
        )""",
        rows,
    )
    conn.commit()
    print(f"Inserted {len(rows)} NOLA restaurants into SQLite")
    conn.close()
    return {r["business_id"] for r in rows}


def ingest_reviews(review_filepath: str, business_ids: set[str]) -> None:
    """Stream review.json, embed NOLA reviews, upsert into ChromaDB."""
    client = chromadb.PersistentClient(path=settings.chroma_path)
    collection = client.get_or_create_collection(
        settings.chroma_collection,
        metadata={"hnsw:M": 16, "hnsw:construction_ef": 100},
    )
    model = EmbeddingModel.from_registry(settings.embed_model)

    start = load_checkpoint()
    batch: dict = {"ids": [], "documents": [], "metadatas": []}
    idx = 0

    for idx, review in enumerate(stream_reviews(review_filepath)):
        if idx < start:
            continue
        if review.get("business_id") not in business_ids:
            continue

        batch["ids"].append(review["review_id"])
        batch["documents"].append(review["text"])
        batch["metadatas"].append({
            "business_id": review["business_id"],
            "stars": float(review["stars"]),
            "date": review["date"],
        })

        if len(batch["ids"]) == BATCH_SIZE:
            _flush_batch(batch, collection, model)
            batch = {"ids": [], "documents": [], "metadatas": []}
            save_checkpoint(idx + 1)
            print(f"Processed review line {idx + 1}")

    if batch["ids"]:
        _flush_batch(batch, collection, model)
        save_checkpoint(idx + 1)
        print(f"Processed review line {idx + 1} (final batch)")


def _flush_batch(batch: dict, collection, model: EmbeddingModel) -> None:
    texts = [EMBED_PREFIX + doc for doc in batch["documents"]]
    embeddings = embed(model, texts)
    collection.upsert(
        ids=batch["ids"],
        embeddings=embeddings,
        documents=batch["documents"],
        metadatas=batch["metadatas"],
    )


# ── Checkpoint helpers ─────────────────────────────────────────────────────────


def load_checkpoint() -> int:
    if CHECKPOINT_FILE.exists():
        return pickle.loads(CHECKPOINT_FILE.read_bytes())
    return 0


def save_checkpoint(idx: int) -> None:
    CHECKPOINT_FILE.write_bytes(pickle.dumps(idx))


# ── Entry point ────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--business", required=True, help="Path to yelp_academic_dataset_business.json")
    parser.add_argument("--review", required=True, help="Path to yelp_academic_dataset_review.json")
    args = parser.parse_args()

    print("Step 1/2: ingesting businesses into SQLite...")
    business_ids = ingest_businesses(args.business)
    print(f"  → {len(business_ids)} NOLA restaurants")

    print("Step 2/2: embedding reviews into ChromaDB...")
    ingest_reviews(args.review, business_ids)
    print("Done.")


if __name__ == "__main__":
    main()
