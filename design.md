# yelp-rag-summarizer — Design Document

## Goal

Build a production-grade REST API that answers queries like *"summarize reviews for
[business] and give me the sentiment breakdown"* using a hierarchical map-reduce RAG
pipeline over 7M Yelp Open Dataset reviews.

The primary deliverable is not just a working API — it is a **benchmark report**
covering retrieval quality, LLM output quality, and system performance.

---

## Approach

**Chosen: Hierarchical Map-Reduce RAG**

Reviews retrieved from ChromaDB → split into batches → **map**: each batch summarized
+ sentiment scored by vLLM in parallel → **reduce**: single LLM call merges partial
summaries into a final coherent summary + aggregated sentiment.

**Approaches not taken:**

- **Simple RAG (Retrieve → Generate):** Single-prompt generation truncates reviews for
  high-volume businesses, degrading summary and sentiment quality. Kept as a baseline
  in benchmarks to quantify the improvement from map-reduce.

- **Agentic RAG:** LLM agent decides what to retrieve (aspect-by-aspect). Rejected
  because agent non-determinism makes consistent benchmarking unreliable, and
  map-reduce already handles scale effectively without the added complexity.

---

## Dataset

**Yelp Open Dataset** — ~7M reviews, ~5GB JSON.

- Large enough to demonstrate genuine large-scale ingestion patterns (streaming,
  batched embedding, idempotent upsert)
- Small enough to process fully on an M4 Pro 24GB without OOM
- Star ratings serve as ground truth for retrieval and sentiment evaluation
- Well-known research benchmark, results are comparable to prior work

---

## Architecture

```
Client
  │
  │  POST /api/v1/analyze
  ▼
FastAPI (async)
  │
  ├── ChromaDB  ←─── retrieval (top-K reviews by business_id)
  │
  ├── LangChain MapReduceDocumentsChain
  │     ├── MAP:    parallel vLLM calls (one per review batch)
  │     └── REDUCE: single vLLM call (merge partial summaries)
  │
  └── LangSmith  ←─── automatic tracing of every LLM call
```

---

## Components

### Ingestion (one-time script)

```
Yelp JSON
  → stream in batches (no full RAM load)
  → chunk reviews per business (max 512 tokens)
  → embed via sentence-transformers (local, CPU)
  → upsert into ChromaDB (persistent, idempotent)
       metadata: business_id, stars, date, category
```

Estimated runtime: ~2-3 hours on M4 Pro for 7M reviews.
Safe to interrupt and re-run — upsert is idempotent.

### API

**Endpoint:** `POST /api/v1/analyze`

Request:
```json
{ "business_id": "abc123", "query": "summarize reviews" }
```

Response:
```json
{
  "summary": "...",
  "sentiment": { "positive": 0.72, "neutral": 0.18, "negative": 0.10 },
  "review_count": 847,
  "latency_ms": 1240
}
```

### Pipeline

1. Retrieve top-K reviews from ChromaDB (K=100–200)
2. Split into batches of N reviews (N=10–20, fits in vLLM context window)
3. MAP: parallel vLLM calls — each batch → partial summary + sentiment
4. REDUCE: single vLLM call — merge partials → final summary + overall sentiment
5. LangSmith traces every step automatically via LangChain callbacks

---

## Tech Stack

| Tool | Role | Justification |
|------|------|---------------|
| **LangChain** | `MapReduceDocumentsChain`, prompt templates, retriever abstraction | Built-in map-reduce, avoids boilerplate |
| **LangSmith** | Tracing every LLM call, latency per step, token usage | Free observability, 2-line integration |
| **ChromaDB** | Vector store for review embeddings, metadata filtering | Persistent, local, Python-native |
| **FastAPI** | Async REST API, Pydantic validation, auto `/docs` | Async needed for parallel map calls |
| **vLLM** | LLM inference — OpenAI-compatible API, PagedAttention batching | Production-grade throughput |
| **sentence-transformers** | Embedding model for ingestion (local, CPU) | Separate from generation, clean separation of concerns |
| **RAGAS** | Primary RAG evaluation framework | Purpose-built for retrieval + generation quality |
| **ROUGE / BERTScore** | Fallback LLM quality metrics | Used when RAGAS LLM-judge calls are too slow |
| **locust** | Load testing for system performance benchmark | Standard, clean p50/p99 reports |

**Models (local on M4 Pro 24GB):**
- Embedding: `all-MiniLM-L6-v2`
- Generation: `Mistral-7B-Instruct` or `Llama-3.1-8B-Instruct` via vLLM

---

## Benchmarks

Three evaluation dimensions:

### 1. Retrieval Quality
- **Tool:** RAGAS — context precision and context recall
- **Ground truth:** Yelp star ratings
- **Method:** Sample 1000 businesses, measure whether retrieved reviews match the
  expected sentiment distribution implied by star ratings

### 2. LLM Output Quality
- **Primary:** RAGAS — faithfulness (no hallucination beyond retrieved reviews),
  answer relevancy (summary answers the query)
- **Fallback:** ROUGE-L and BERTScore against held-out reference summaries
- **Baseline:** Compare map-reduce output vs. simple RAG (Approach A) on same samples

### 3. System Performance
- **Tool:** locust async load test
- **Metrics:** p50 / p99 latency, requests/sec, error rate under load
- **Target:** p99 < 5s for businesses with up to 500 reviews

Results published in `README.md` as a comparison table.

---

## Error Handling

| Scenario | Behaviour |
|----------|-----------|
| ChromaDB returns 0 results | 404 with descriptive message |
| vLLM timeout on map step | Fail entire request — no silent partial results |
| Ingestion interrupted | Safe to re-run — ChromaDB upsert is idempotent |
| RAGAS LLM-judge fails | Automatically fall back to ROUGE/BERTScore |

---

## Project Structure

```
yelp-rag-summarizer/
├── ingestion/
│   └── ingest.py          # stream → embed → ChromaDB
├── api/
│   ├── main.py            # FastAPI app and routes
│   ├── pipeline.py        # LangChain MapReduce chain
│   ├── retriever.py       # ChromaDB retrieval logic
│   └── schemas.py         # Pydantic request/response models
├── benchmarks/
│   ├── retrieval.py       # RAGAS context precision/recall
│   ├── llm_quality.py     # RAGAS faithfulness + relevancy; ROUGE/BERTScore fallback
│   └── load_test.py       # locust load test
├── config.py              # vLLM endpoint, ChromaDB path, model names
├── design.md              # this file
├── requirements.txt
└── README.md              # benchmark results table
```
