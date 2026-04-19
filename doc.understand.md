# Semantic Search — How It Works

## What the code does

1. **Load ChromaDB** — connects to the persistent store and gets the `yelp_reviews` collection.
2. **Embed the query** — encodes the query string using `nomic-embed-text-v1.5` via MLX, then applies the same MRL truncation + L2 renormalisation used at ingestion time (slice to 256 dims, divide by norm).
3. **Query ChromaDB** — calls `collection.query()` with the query embedding, filtered to a single `business_id`. Returns top-N results ranked by cosine distance.

## Key distinction from `retrieve_reviews()`

`retriever.py` uses `collection.get()` — a metadata-only fetch, no ranking. The pipeline will use `collection.query()` with an embedded query to get semantically ranked results.

## Experiment: query `biz_002` for "omakase"

| Rank | Stars | Distance | Review |
|------|-------|----------|--------|
| 1 | 4★ | 0.72 | "The omakase is pricey but worth every dollar. Chef clearly sources premium fish." |
| 2 | 5★ | 0.95 | "Best ramen outside Japan. Rich tonkotsu broth, perfectly chewy noodles..." |
| 3 | 4★ | 1.00 | "Excellent karaage chicken and the house sake selection is impressive." |
| 4 | 3★ | 1.04 | "Good but overhyped. The ramen broth was fine, not revelatory." |
| 5 | 5★ | 1.07 | "Perfect unhurried meal. The black sesame ice cream is not to be missed." |

**Distance gap:** rank 1 is at 0.72, rank 2 jumps to 0.95 — clear separation for the direct hit. Ranks 2–5 cluster around Japanese dining concepts, which is semantically correct even though none mention omakase.

## Embedding consistency rule

Query embeddings **must** use the same truncation + renormalisation as ingestion. Skipping either step produces distances that are not comparable to stored vectors.
