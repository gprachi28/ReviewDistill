# yelp-rag-summarizer — Design Document

## Goal

Build a production-grade REST API over 7M Yelp Open Dataset reviews that returns
sentiment analysis and business summarization via a RAG pipeline.

The primary deliverable is not just a working API — it is a **benchmark report**
covering retrieval quality, LLM output quality, and system performance.

The project is built in three versioned stages, each benchmarked independently so
the improvement from each stage is measurable.

---

## Versioned Approach

### v1 — Simple RAG (sentiment only)
Retrieve top-K reviews from ChromaDB → single vLLM call → sentiment breakdown.
No summarization. Goal: get the full stack working end-to-end with clean benchmarks.

### v2 — Two-Stage Aggregation (summary + sentiment)
Retrieve top 50 reviews → filter to 20 most signal-rich reviews using star-rating
metadata (highest and lowest stars carry the most information) → single vLLM call
with full 20-review context → summary + sentiment.

Uses Llama-3.1-8B-Instruct (128K context window), so 20 reviews (~6,000 tokens)
fits comfortably in one call. No partial JSON merging, no parallel coordination.
Removes map-reduce complexity while still producing a summarized output.

### v3 — Hierarchical Map-Reduce (full scale)
Retrieve top 100–200 reviews → split into batches → parallel vLLM map calls →
single reduce call merges partial summaries. Built using LCEL (not the deprecated
`MapReduceDocumentsChain`). Handles businesses with thousands of reviews without
truncation. Benchmarked against v2 to quantify the quality improvement.

**Approach not taken:**

- **Agentic RAG:** LLM agent decides what to retrieve (aspect-by-aspect). Rejected
  because agent non-determinism makes consistent benchmarking unreliable, and
  the v2/v3 progression already handles scale effectively without the added complexity.

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
  ├── ChromaDB  <─── retrieval (top-K reviews by business_id)
  │
  ├── LangChain pipeline (LCEL)
  │     v1: retrieve → single LLM call
  │     v2: retrieve → metadata filter → single LLM call
  │     v3: retrieve → batch → parallel map → reduce
  │
  └── LangSmith  <─── automatic tracing of every LLM call
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

Estimated runtime: 10–20 hours on M4 Pro for 7M reviews (CPU embedding).
Ingestion script checkpoints progress after every batch — safe to interrupt and resume.

### API

**Endpoint:** `POST /api/v1/analyze`

Request:
```json
{ "business_id": "abc123" }
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

v1 returns `sentiment` only (no `summary` field). v2 and v3 return both.

### Pipeline per version

**v1:**
1. Retrieve top-K reviews from ChromaDB (K=50)
2. Single vLLM call → sentiment breakdown (JSON mode enforced)
3. LangSmith traces the call

**v2:**
1. Retrieve top 50 reviews from ChromaDB
2. Filter to 20 most signal-rich using star-rating metadata (top 10 highest + top 10 lowest rated)
3. Single vLLM call with all 20 reviews in context → summary + sentiment (JSON mode enforced)
4. LangSmith traces the call

**v3:**
1. Retrieve top 100–200 reviews from ChromaDB
2. Split into batches of 10–20 reviews
3. MAP: parallel async vLLM calls — each batch → partial summary + sentiment
4. REDUCE: single vLLM call — merge partials → final summary + aggregated sentiment
5. LangSmith traces every step

---

## Tech Stack

| Tool | Role | Justification |
|------|------|---------------|
| **LangChain (LCEL)** | Pipeline orchestration, prompt templates, retriever abstraction | LCEL is the current non-deprecated API |
| **LangSmith** | Tracing every LLM call, latency per step, token usage | Free observability, 2-line integration |
| **ChromaDB** | Vector store for review embeddings, metadata filtering | Persistent, local, Python-native |
| **FastAPI** | Async REST API, Pydantic validation, auto `/docs` | Async needed for parallel map calls in v3 |
| **vLLM** | LLM inference — OpenAI-compatible API, JSON mode, PagedAttention | Production-grade throughput; cloud GPU for benchmarks |
| **sentence-transformers** | Embedding model for ingestion (local, CPU) | Separate from generation; clean separation of concerns |
| **RAGAS** | Primary RAG evaluation framework | Purpose-built for retrieval + generation quality |
| **ROUGE / BERTScore** | Fallback LLM quality metrics | Used when RAGAS LLM-judge calls are too slow |
| **locust** | Load testing for system performance benchmark | Standard, clean p50/p99 reports |

**Models:**
- Embedding: `all-MiniLM-L6-v2` (384-dim, CPU, ingestion only)
- Generation: `Llama-3.1-8B-Instruct` via vLLM (128K context window)
- Dev/local: `mlx-lm` or `llama.cpp` (M4 Pro Metal backend, no CUDA required)
- Benchmark runs: vLLM on cloud GPU (Lambda Labs / RunPod A10G)

---

## Benchmarks

Three evaluation dimensions, run at each version (v1, v2, v3) to show progression:

### 1. Retrieval Quality
- **Tool:** RAGAS — context precision
- **Ground truth:** Yelp star ratings
- **Method:** Sample 1000 businesses, measure whether retrieved reviews match the
  expected sentiment distribution implied by star ratings
- Note: context recall skipped — Yelp has no reference summaries

### 2. LLM Output Quality
- **Primary:** RAGAS — faithfulness (no hallucination), answer relevancy
- **Fallback:** ROUGE-L and BERTScore (when RAGAS LLM-judge is too slow)
- **Baseline comparison:** v1 vs v2 vs v3 on the same 100-business sample
- Note: star-vs-LLM sentiment divergence reported as a finding, not an error
  (3-star reviews are often bimodal, not neutral — conflicts are interesting)

### 3. System Performance
- **Tool:** locust async load test
- **Metrics:** p50 / p99 latency, requests/sec, error rate under load
- **Targets:**
  - v1: p99 < 3s (local M4 Pro), p99 < 1s (cloud GPU)
  - v2: p99 < 8s (local M4 Pro), p99 < 2s (cloud GPU)
  - v3: p99 < 20s (local M4 Pro), p99 < 5s (cloud GPU)

Results published in `README.md` as a version comparison table.

---

## Error Handling

| Scenario | Behaviour |
|----------|-----------|
| ChromaDB returns 0 results | 404 with descriptive message |
| vLLM timeout on any LLM call | Fail entire request — no silent partial results |
| Ingestion interrupted | Safe to re-run — checkpoint file + idempotent upsert |
| RAGAS LLM-judge fails | Automatically fall back to ROUGE/BERTScore |
| v3 map step returns malformed JSON | Discard that batch, continue reduce with remaining |

---

## Known Pitfalls & Mitigations

### 1. vLLM Does Not Officially Support Apple Silicon
vLLM is built for CUDA. M4 Pro has Metal/MPS — experimental support is unstable and
may silently fall back to CPU, destroying throughput benchmarks.

**Mitigation:** Use `mlx-lm` or `llama.cpp` locally for development. Run vLLM on a
cloud GPU (Lambda Labs, RunPod) for benchmark runs only. README documents this split.

---

### 2. Memory Pressure: vLLM + ChromaDB on 24GB
A 7B model in vLLM (~14GB) + ChromaDB HNSW index for 7M 384-dim embeddings (~10GB)
leaves near-zero headroom on 24GB unified memory.

**Mitigation:**
- Use `all-MiniLM-L6-v2` (384-dim) to keep the vector index to ~10GB
- Tune ChromaDB HNSW parameters (`M=16`, `ef_construction=100`) for memory efficiency
- Do not run vLLM and ingestion simultaneously — ingest first, then start vLLM
- Consider mmap mode for ChromaDB if memory pressure spikes during query serving

---

### 3. Ingestion Runtime Is 10–20 Hours, Not 2–3
Embedding 7M reviews at ~100–200 reviews/sec on CPU takes far longer than expected.

**Mitigation:** Ingest script checkpoints after every batch (last processed index
saved to disk). Restart resumes from that index. ChromaDB upsert is idempotent.

---

### 4. Structured Sentiment Output Requires Enforced JSON Mode
Getting `{ "positive": 0.72, "neutral": 0.18, "negative": 0.10 }` reliably from a
generative model requires enforced structured output — prompt-only instructions are
not reliable enough for production use.

**Mitigation:** Use vLLM's guided decoding (outlines / JSON mode) with a Pydantic
schema enforced at inference time across all versions.

---

### 5. v2 Filter Step: Star Ratings Are Not Neutral at 3 Stars
A 3-star review is often bimodal (strong positives + strong negatives). Filtering
purely by "extreme stars" may miss nuanced but information-rich mid-range reviews.

**Mitigation:** Filter to top 10 highest + top 10 lowest rated reviews. Do not
include mid-range reviews in the filtered set for v2. Star-vs-LLM divergence is
reported as a benchmark finding, not an accuracy failure.

---

### 6. v3 Map-Reduce Semantic Drift
The reduce step is only as good as the map step. Without strict instructions,
intermediate summaries lose specific entities — "cold steak" becomes "mixed feelings
about food temperature." Each reduce step compounds the loss.

**Mitigation:** Map prompt must explicitly instruct: *"preserve all specific nouns,
product names, and verbatim complaints."* Measure semantic drift using RAGAS
faithfulness — this is a reportable comparison between v2 and v3.

---

### 7. p99 Latency Targets Are Hardware-Dependent
Local M4 Pro numbers will be significantly worse than cloud GPU numbers.

**Mitigation:** Report both explicitly. This turns a hardware limitation into a
benchmark result showing the value of proper inference infrastructure.

---

### 8. RAGAS Context Recall Requires Reference Answers Yelp Does Not Provide
**Mitigation:** Skip context recall. Use context precision, faithfulness, and answer
relevancy only — all three are reference-free metrics.

---

### 9. Yelp Dataset Access Requires Manual Registration
The dataset is not pip-installable — requires manual download from Yelp's website.

**Mitigation:** Document exact download steps in README. Include a 100-review fixture
in the repo for CI and local smoke tests that do not require the full dataset.

---

## Project Structure

```
yelp-rag-summarizer/
├── ingestion/
│   └── ingest.py              # stream → embed → ChromaDB, with checkpointing
├── api/
│   ├── main.py                # FastAPI app and routes
│   ├── pipeline_v1.py         # simple RAG: retrieve → single LLM call
│   ├── pipeline_v2.py         # two-stage: retrieve → metadata filter → single LLM call
│   ├── pipeline_v3.py         # map-reduce: retrieve → batch → parallel map → reduce
│   ├── retriever.py           # ChromaDB retrieval logic (shared)
│   └── schemas.py             # Pydantic request/response models (shared)
├── benchmarks/
│   ├── retrieval.py           # RAGAS context precision
│   ├── llm_quality.py         # RAGAS faithfulness + relevancy; ROUGE/BERTScore fallback
│   └── load_test.py           # locust load test
├── fixtures/
│   └── sample_reviews.json    # 100-review subset for CI and smoke tests
├── config.py                  # vLLM endpoint, ChromaDB path, model names
├── design.md                  # this file
├── requirements.txt
└── README.md                  # benchmark results comparison table (v1 vs v2 vs v3)
```
