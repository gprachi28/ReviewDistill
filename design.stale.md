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
  → embed via mlx-embedding-models (MPS, Apple Silicon GPU)
  → upsert into ChromaDB (persistent, idempotent)
       metadata: business_id, stars, date, category
```

Estimated runtime: 2–4 hours on M4 Pro for 7M reviews (MPS embedding via mlx-embedding-models).
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
| **vllm-metal** | LLM inference — OpenAI-compatible API, JSON mode, PagedAttention | Apple Silicon port of vLLM ([github.com/vllm-project/vllm-metal](https://github.com/vllm-project/vllm-metal)); installed from source, not pip |
| **openai SDK** | HTTP client for vLLM's `/v1` endpoint | vLLM exposes an OpenAI-compatible REST API; the `openai` SDK gives structured output support, retries, and type safety at no extra cost — `api_key="not-needed"` since there is no OpenAI involvement |
| **mlx-embedding-models** | Embedding model for ingestion (MPS, Apple Silicon) | Runs `nomic-embed-text-v1.5` on M4 Pro GPU via MLX; significantly faster than CPU |
| **RAGAS** | Primary RAG evaluation framework | Purpose-built for retrieval + generation quality |
| **ROUGE / BERTScore** | Fallback LLM quality metrics | Used when RAGAS LLM-judge calls are too slow |
| **locust** | Load testing for system performance benchmark | Standard, clean p50/p99 reports |

**Models:**
- Embedding: `nomic-embed-text-v1.5` (256-dim via Matryoshka truncation, MPS, ingestion only)
- Generation: `Llama-3.1-8B-Instruct` via `vllm-metal` (M4 Pro) and vLLM (cloud GPU)
- Benchmark runs: both M4 Pro (Metal) and cloud GPU (Lambda Labs / RunPod A10G)
  to separate hardware effects from pipeline quality differences

**vllm-metal rationale:**

vLLM was chosen over alternatives (Ollama, llama.cpp, Hugging Face `pipeline`) because:
- It exposes an OpenAI-compatible REST API out of the box, so the client code and cloud GPU code are identical — no code changes when switching from M4 Pro to a Lambda Labs A10G for benchmarks.
- JSON mode (guided decoding via outlines) enforces the structured sentiment schema at inference time, which prompt-only instructions cannot guarantee reliably.
- PagedAttention handles concurrent requests cleanly, which matters for the locust load test in Phase 5.

`vllm-metal` specifically is used on Apple Silicon because the official `vllm` package targets CUDA only. It is installed from source ([github.com/vllm-project/vllm-metal](https://github.com/vllm-project/vllm-metal)) and is intentionally absent from `requirements.txt` — see Known Pitfall #1 for install steps.

**OpenAI SDK rationale:**

The `openai` Python SDK is used as the HTTP client pointed at vLLM's local `/v1` endpoint (`base_url="http://localhost:8000/v1"`, `api_key="not-needed"`). Alternatives considered:

- **Raw `httpx`**: works but loses structured output helpers, type hints, and retry logic.
- **LangChain `ChatOpenAI`**: same wire protocol; would integrate with LangSmith tracing automatically. A viable swap if the pipeline is refactored to use LCEL chains throughout — noted as a future improvement.
- **LiteLLM**: useful for abstracting over multiple providers but adds an unnecessary dependency given we only target vLLM.

The `openai` SDK was chosen because it is the lightest path to a working client with JSON mode support, and vLLM's compatibility layer makes it transparent.

**Embedding model design rationale:**

Candidates considered:

| Model | Dim | Index size (7M rows) |
|-------|-----|----------------------|
| `bge-small-en-v1.5` | 384 | ~10GB |
| `all-mpnet-base-v2` | 768 | ~20GB |
| `nomic-embed-text-v1.5` ✓ | 256 (MRL) | ~7GB |

- `bge-small-en-v1.5`: better than MiniLM but fixed 384-dim — no way to trade quality for memory
- `all-mpnet-base-v2`: strongest quality at 768-dim but ~20GB index leaves no room for vLLM on 24GB
- `nomic-embed-text-v1.5` at 256-dim: MRL means the 256-dim subspace is explicitly optimised during training, not naively truncated — retrieval quality holds, index drops to ~7GB, and the model runs on MPS via `mlx-embedding-models`

**Generation model design rationale:**

`Llama-3.1-8B-Instruct` was chosen because it fits the 24GB memory budget (~14GB) alongside the ChromaDB index (~7GB), has a 128K context window required for v2 and v3, and has reliable JSON-mode compliance for structured sentiment output.

Alternative generation models considered:

| Model | Params | Context | Notes |
|-------|--------|---------|-------|
| `Qwen2.5-7B-Instruct` | 7B | 128K | Strong structured output, benchmarks very close to Llama-3.1-8B on instruction tasks, slightly smaller so saves ~1-2GB |
| `Phi-3.5-mini-instruct` | 3.8B | 128K | Half the memory (~7GB) — frees significant headroom. Weaker summarization than Llama but may be enough for sentiment |
| `Gemma-2-9B-Instruct` | 9B | 8K | Strong quality but 8K context is a problem — v2 with 20 reviews (~6K tokens) would be right at the edge, v3 impossible |
| `Mistral-7B-Instruct-v0.3` | 7B | 32K | Solid, but 32K context is tight for v3 (100-200 retrieved reviews) |

**Backup:** `Qwen2.5-7B-Instruct` is the closest like-for-like swap — same 128K context, slightly smaller memory footprint, and structured output compliance is marginally stronger on benchmarks. If Llama-3.1-8B-Instruct causes issues (Metal instability, JSON-mode failures, memory pressure), swap to Qwen2.5-7B-Instruct first and re-run the same RAGAS benchmarks. Any quality difference becomes a reportable finding.

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

### 1. vLLM on Apple Silicon Requires the Metal Port
The official vLLM repo targets CUDA. Apple Silicon support is provided via
`vllm-metal` (https://github.com/vllm-project/vllm-metal), a port maintained under
the vllm-project org. It may lag behind the main repo on features or have
Metal-specific bugs not present in the CUDA build.

**Mitigation:** Install via the official install script (not `git clone && pip install -e .`):
```bash
curl -fsSL https://raw.githubusercontent.com/vllm-project/vllm-metal/main/install.sh | bash
```
This installs vllm-metal and the vLLM core into a dedicated venv at `~/.venv-vllm-metal`.
Activate it before running `vllm serve`:
```bash
source ~/.venv-vllm-metal/bin/activate
```
Run benchmark comparisons on both M4 Pro (Metal) and a cloud GPU (Lambda Labs / RunPod A10G)
to separate hardware effects from pipeline effects.

---

### 2. Memory Pressure: vLLM + ChromaDB on 24GB
A 7B model in vLLM (~14GB) + ChromaDB HNSW index for 7M 256-dim embeddings (~7GB)
leaves ~3GB headroom on 24GB unified memory.

**Mitigation:**
- Use `nomic-embed-text-v1.5` truncated to 256-dim via Matryoshka Representation Learning
  (MRL) — same model, lower-dimensional projection, ~7GB index vs ~10GB for 384-dim
- Tune ChromaDB HNSW parameters (`M=16`, `ef_construction=100`) for memory efficiency
- Do not run vLLM and ingestion simultaneously — ingest first, then start vLLM
- Consider mmap mode for ChromaDB if memory pressure spikes during query serving

---

### 3. Ingestion Runtime: Estimate for MPS vs CPU
Embedding 7M reviews on CPU at ~100–200 reviews/sec takes 10–20 hours. Using
`mlx-embedding-models` on the M4 Pro MPS GPU targets ~2–4 hours at ~500–1000 reviews/sec.

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
