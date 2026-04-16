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

## Known Pitfalls & Mitigations

### 1. vLLM Does Not Officially Support Apple Silicon
vLLM is built for CUDA. M4 Pro has Metal/MPS — experimental support is unstable and
may silently fall back to CPU, destroying throughput benchmarks.

**Mitigation:** Run vLLM on a cheap cloud GPU (Lambda Labs, RunPod) for benchmark
runs. Use `mlx-lm` or `llama.cpp` locally for development iteration. README must
document that vLLM is the production deployment target, not the local dev backend.

---

### 2. Memory Pressure: vLLM + ChromaDB on 24GB
A 7B model in vLLM (~14GB) + ChromaDB HNSW index for 7M 384-dim embeddings (~10GB)
leaves near-zero headroom on 24GB unified memory.

**Mitigation:**
- Use `all-MiniLM-L6-v2` (384-dim, not 768-dim) to keep the vector index to ~10GB
- Tune ChromaDB HNSW parameters (`M=16`, `ef_construction=100`) for memory efficiency
- Do not run vLLM and ChromaDB ingestion simultaneously — ingest first, then start vLLM
- If memory pressure spikes during query serving, consider mmap mode for ChromaDB

---

### 3. Ingestion Runtime Far Exceeds Initial Estimate
Embedding 7M reviews at ~100-200 reviews/sec (CPU) takes 10–20 hours, not 2-3.
An interrupted run without checkpointing means re-embedding from scratch.

**Mitigation:** Ingest script saves last processed index to a checkpoint file after
every batch. On restart, reads checkpoint and resumes from that position. ChromaDB
upsert is idempotent — safe to overlap.

---

### 4. LangChain `MapReduceDocumentsChain` Is Being Deprecated
The old chain API is losing maintenance in favour of LCEL (LangChain Expression
Language). Deprecation warnings will clutter LangSmith traces.

**Mitigation:** Build the map-reduce pipeline using LCEL from the start.

---

### 5. Structured Sentiment Output from a Generative LLM
Getting `{ "positive": 0.72, "neutral": 0.18, "negative": 0.10 }` reliably from a
generative model requires enforced structured output. Without it, the model
occasionally returns free text or misformatted numbers, breaking the API schema.

**Mitigation:** Use vLLM's guided decoding (outlines / JSON mode) with a Pydantic
schema enforced at inference time. Never rely on prompt-only instructions for
structured numeric output.

---

### 6. Map-Reduce Semantic Drift ("Hallucination Loop")
The reduce step is only as good as the map step. Without strict instructions,
intermediate summaries lose specific entities and complaints — e.g., "cold steak"
becomes "mixed feelings about food temperature." Each reduce step compounds the loss.

**Mitigation:** Map prompt must explicitly instruct: *"preserve all specific nouns,
product names, and verbatim complaints."* Measure semantic drift using RAGAS
faithfulness between map outputs and source reviews — this is a reportable finding.

---

### 7. Star Rating ≠ Sentiment Ground Truth
A 3-star Yelp review is often bimodal (strong positives + strong negatives), not
neutral. Treating star ratings as a direct sentiment proxy will produce misleading
evaluation numbers.

**Mitigation:** Do not average stars as a sentiment score. Instead, treat
star-vs-LLM-sentiment divergence as a **benchmark finding**: identify businesses
where the two conflict and explain why (bimodal reviews, single-issue complaints,
etc.). This makes the evaluation section more interesting than a simple accuracy number.

---

### 8. p99 < 5s Latency Target Is Unrealistic on Local Hardware
With 100-200 reviews per query and 10+ LLM calls in the map-reduce chain, p99 < 5s
is achievable on a cloud GPU but not on an M4 Pro.

**Mitigation:** Report two numbers — local (M4 Pro, expected p99 ~15s) and cloud
(A10G/A100, expected p99 ~3-4s). This turns a limitation into a benchmark result.
Adjust the official target: **p99 < 5s on cloud GPU; p99 < 20s on M4 Pro.**

---

### 9. RAGAS Context Recall Requires Reference Answers
RAGAS context recall needs a ground-truth answer to compare against. Yelp does not
ship human summaries.

**Mitigation:** Limit RAGAS evaluation to metrics that do not require ground-truth
answers: context precision, faithfulness, and answer relevancy. Skip context recall,
or generate reference summaries for a small held-out set (100 businesses) manually.

---

### 10. Yelp Dataset Access Requires Manual Registration
The Yelp Open Dataset is not pip-installable. It requires manual registration, terms
acceptance, and a manual download from Yelp's website.

**Mitigation:** Document the exact download steps in README. Provide a small sample
fixture (100 reviews) in the repo for CI and local smoke tests that do not require
the full dataset.

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
