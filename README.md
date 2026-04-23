# Yelp Conversational Assistant

A conversational assistant that answers natural language questions about New Orleans
restaurants. Ask "Find a jazz brunch spot with outdoor seating that feels festive, not
touristy" — get named recommendations with review-grounded reasoning.

---

## How It Works

Queries are answered in two steps:

**Step 1 — SQL filter (structured precision).** A Query Planner LLM parses the natural
language question into structured filters drawn from Yelp business attributes (noise level,
price range, outdoor seating, live music, brunch-friendly, etc.). This shrinks 1,199 New
Orleans restaurants down to a precise candidate pool with 100% metadata accuracy.

**Step 2 — Semantic search (review-text ranking).** Within that candidate pool, vector
similarity search over ~621K embedded review snippets ranks results by what reviewers
actually wrote — food quality, atmosphere nuance, experience narratives, local knowledge.
A Synthesizer LLM turns the top review evidence into a conversational answer.

```
User question
      │
      ▼
 Query Planner (LLM)          → {sql_filters, semantic_query, intent}
      │
  ┌───┴───┐
  ▼       ▼
SQL     Semantic
filter  search (ChromaDB)
  │       │
  └───┬───┘
      ▼
 Synthesizer (LLM)            → conversational answer + named recommendations
```

---

## Dataset

**Source:** Yelp Open Dataset — `business.json` + `review.json`

**Subset:** New Orleans restaurants with `review_count > 50`

| Stat | Value |
|---|---|
| Businesses | 1,199 |
| Reviews | ~621K |
| Ingest time | ~30–45 min on M4 Pro (MPS) |

New Orleans was chosen for its semantically rich review vocabulary — Creole, Cajun,
po'boys, beignets, jazz brunch, second lines, Frenchmen Street — giving the semantic
retrieval real signal to work with.

---

## Architecture

### API

| Version | Description |
|---|---|
| **v1** | Single-shot: each request is independent, no session state |
| **v2** | Multi-turn: session-aware, last 3 turns injected into Query Planner |

### SQL filter fields (from Yelp business attributes)

Noise level, price range, alcohol, outdoor seating, good for groups, reservations, good
for kids, good for meal (brunch/latenight/etc.), live music, happy hour, wheelchair
accessible, dogs allowed, ambience (romantic/classy/hipster/etc.), parking, attire, WiFi.

### Semantic query

Runs over embedded review text — captures what no structured attribute can: specific
dishes, nuanced atmosphere, experience narratives, local knowledge.

---

## Security

The Query Planner is an LLM — its output is untrusted user-controlled data at the SQL
boundary.

**SQL injection prevention:** `sql_filter.py` validates all LLM-generated field names
against explicit whitelists (`_BOOLEAN_FIELDS`, `_SCALAR_FIELDS`, `_JSON_FIELDS`) before
any interpolation into SQL strings. JSON attribute sub-keys (e.g. `brunch`, `live`) are
checked against a per-field allowlist. Any unrecognised field or sub-key is silently
skipped. Filter values are always bound via SQLite parameterised queries — LLM output
never touches a SQL string directly.

**Input validation:** `question` fields on all API request schemas enforce
`max_length=2000` to prevent oversized inputs from exhausting LLM context.

**Session IDs:** Generated server-side with `uuid.uuid4()`. Clients supply an ID to
continue a session; unknown IDs start fresh rather than erroring.

---

## Stack

| Tool | Role |
|---|---|
| FastAPI | REST API |
| SQLite | Business metadata + attribute filtering |
| ChromaDB | Vector store for review embeddings (persistent, local) |
| nomic-embed-text-v1.5 | 256-dim MRL embeddings via mlx-embedding-models (MPS) |
| Qwen2.5-7B-Instruct | Query Planner + Synthesizer via vllm-metal |
| openai SDK | HTTP client pointed at vLLM's `/v1` endpoint |
| pydantic-settings | Config management via `.env` |

---

## Setup

**Requirements:** Python 3.11+, Apple Silicon M-series

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

**vllm-metal** (Apple Silicon):
```bash
curl -fsSL https://raw.githubusercontent.com/vllm-project/vllm-metal/main/install.sh | bash
source ~/.venv-vllm-metal/bin/activate
```

**Environment:**
```bash
cp .env.example .env
# fill in VLLM_BASE_URL
```

---

## Running

**1. Ingest** (one-time, ~30–45 min on M4 Pro):
```bash
python -m ingestion.ingest_nola \
  --business-file /path/to/yelp_academic_dataset_business.json \
  --review-file /path/to/yelp_academic_dataset_review.json
```

**2. Start vLLM server:**
```bash
vllm serve Qwen/Qwen2.5-7B-Instruct \
  --port 8001 \
  --max-model-len 16384 \
  --gpu-memory-utilization 0.95
```

**3. Start API:**
```bash
uvicorn api.main:app --port 8000
```

**4. Query:**
```bash
curl -X POST http://localhost:8000/api/v1/query \
  -H "Content-Type: application/json" \
  -d '{"question": "loud spot for a bachelor party that handles large groups"}'
```

---

## Benchmarks

_Results to be filled after evaluation runs._

| Metric | v1 | v2 |
|---|---|---|
| Query planning accuracy (20-query eval set) | — | — |
| RAGAS faithfulness | — | — |
| p50 latency (M4 Pro, warm) | — | — |
| p99 latency (M4 Pro, warm) | — | — |
| req/s (5 concurrent users) | — | — |
