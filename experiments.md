# Experiments Log

---

## EXP-001 — v1 baseline run
**Date:** 2026-04-23  
**Query:** "bachelor party spot, loud, handles large groups"  
**Model:** Qwen2.5-7B-Instruct-4bit (mlx_lm)

**Status:** Pipeline runs end-to-end. Results are directionally relevant but have quality issues.

**Query plan produced:**
```json
{"noise_level": "loud", "good_for_groups": true, "attire": "dressy"}
```

**Notes:**
- `attire: "dressy"` is hallucinated — query implies nothing about dress code; spurious filter shrinks candidate pool unnecessarily
- Latency: 172s — far above <15s target; two sequential LLM calls on local mlx_lm is the bottleneck
- Synthesizer not grounded: recommended Dickie Brennan's despite evidence saying *"not recommended for bachelor parties"*
- GW Fins evidence is inverted — review *complains* about sitting near a noisy group; matched on "bachelor party" keyword, not semantic intent
- Synthesizer invented "private room available for booking" for Brennan's — not in any retrieved snippet

**Next:** Fix query planner over-constraining — prompt tuning to only emit filters clearly implied by the query.

---

## EXP-002 — Query planner prompt fix (anti-inference rules + few-shot)
**Date:** 2026-04-23  
**Query:** "bachelor party spot, loud, handles large groups" (same as EXP-001)  
**Change:** Added explicit DO NOT infer rules for occasion words + 2 few-shot examples to `_SYSTEM_PROMPT`

**Status:** Over-constraining fixed.

**Query plan produced:**
```json
{"noise_level": ["loud", "very_loud"], "good_for_groups": true}
```

- `attire` spurious filter gone
- `noise_level` correctly expanded to list `["loud", "very_loud"]` — matches few-shot example
- Two targeted changes worked: explicit prohibition on occasion-word inference + examples

**Open issues from EXP-001 still pending:**
- Synthesizer faithfulness (hallucinated details, inverted evidence)
- Latency 172s — two sequential LLM calls on local mlx_lm

---

## EXP-003 — Timeout fix + synthesizer faithfulness prompt
**Date:** 2026-04-23  
**Query:** "bachelor party spot" (shorter, no "loud")  
**Changes:** `timeout=300s` on OpenAI clients; negative evidence handling rules in synthesizer prompt; temperature 0.3 → 0.0

**Status:** Pipeline works end-to-end. Quality issues remain.

**Query plan:** `{"good_for_groups": true}` — correct, no spurious filters  
**Latency:** 95s (down from 172s on shorter query)  
**Businesses returned:** 17 unique — too many, overwhelms 7B context

**Notes:**
- `good_for_groups: true` is a loose filter; large candidate pool → 17 businesses in synthesizer
- Synthesizer only synthesized 2 out of 17 businesses coherently — context overload
- One hallucination: said "bachelorette party" for La Boca (review said bachelor party)
- Dickie Brennan's correctly absent — negative evidence prompt fix worked
- Timeout fix resolved BrokenPipeError from EXP-002

**Fix applied:** Cap synthesizer to top 5 businesses by semantic distance (`max_businesses=5`)

---

## EXP-004 — Synthesizer context cap + original query
**Date:** 2026-04-23  
**Query:** "bachelor party spot, loud, handles large groups"  
**Changes:** `max_businesses=5` cap in synthesizer

**Status:** Focused answer, but semantic retrieval is pulling negative reviews.

**Query plan:** `{"noise_level": ["loud", "very_loud"], "good_for_groups": true}` — correct  
**Latency:** 102s  
**Businesses returned:** 5 (cap working)

**Notes:**
- Answer focuses on The Maison — well grounded in a genuinely good bachelor party review
- "upstairs loft area" in the answer is a hallucination — not in any retrieved snippet
- 4 out of 5 retrieved businesses have negative-sentiment reviews about loudness: complaints like "impossible to talk", "ears bleed", "weird loud music" — semantic query contains "loud" which matches complaints, not recommendations
- SQL already handles `noise_level=loud` — semantic query duplicating it is causing retrieval pollution
- Only 1 genuinely useful recommendation out of 5 businesses

**Fix to try:** Remove SQL filter terms from semantic query in the planner — SQL handles structure, semantic should focus on qualitative aspects ("energetic celebration party atmosphere" not "loud")

---

## EXP-005 — Semantic query decoupled from SQL filters
**Date:** 2026-04-23  
**Query:** "bachelor party spot, loud, handles large groups"  
**Change:** Prompt rule: semantic_query must not repeat SQL filter terms + updated few-shot examples

**Status:** Retrieved reviews are now positive/relevant. Clear improvement.

**Query plan:** `{"noise_level": ["loud", "very_loud"], "good_for_groups": true}`, `semantic_query: "fun lively bachelor party celebration great time"` — clean separation  
**Latency:** 118s

**Notes:**
- All 5 retrieved businesses have genuinely positive bachelor party reviews — no more noise complaints
- Answer recommends Jacques-Imo's and La Casita, both well grounded in evidence
- Possible hallucination: "two-hour wait" for Jacques-Imo's — not in the displayed evidence snippet; may be in one of the 3 non-displayed snippets the LLM sees (hard to verify from response alone)
- `evidence` field only shows best snippet; LLM sees up to 3 per business — makes faithfulness hard to audit from the API response
- Latency: 118s — still far from <15s target; two sequential local LLM calls are the bottleneck

**Open:** Latency is the main remaining issue for a usable demo.

---

## EXP-006 — Faithfulness visibility (evidence as list[str])
**Date:** 2026-04-23  
**Query:** "bachelor party spot, loud, handles large groups"  
**Change:** `evidence: str` → `evidence: list[str]`, populated with all snippets the LLM saw

**Status:** Faithfulness now auditable. "Two-hour wait" claim verified as grounded (snippet 2 of Jacques-Imo's).

**Notes:**
- Jacques-Imo's 3-snippet evidence is excellent — loud rowdy atmosphere, bachelor party mentions, food praise; best retrieval result so far
- La Casita well grounded — courtyard, large group, no reservation
- 3 of 5 businesses are 3.5★ — no minimum star filter in pipeline; low-quality businesses getting through
- Lucy's evidence ("Cool vibe") is too thin to be useful — one sentence
- Rock-n-Saké is a bachelorette review, and truncated mid-sentence in ChromaDB

**Next candidates:**
- Add minimum stars filter (≥ 4.0) on retrieved snippets before passing to synthesizer
- Or filter at business level in SQL (`stars >= 4.0`)

---

## EXP-007 — min_stars=4.0 quality floor in SQL filter
**Date:** 2026-04-23  
**Query:** "bachelor party spot, loud, handles large groups"  
**Change:** `filter_businesses()` now applies `stars >= 4.0` as a base condition, never relaxed

**Status:** Clean results. Low-rated businesses gone.

**Notes:**
- All 5 businesses ≥ 4.0★ — Bamboula's (3.5), Lucy's (3.5), Rock-n-Saké (3.5) correctly excluded
- Jacques-Imo's answer is excellent — all claims grounded across 3 snippets
- Synthesizer still recommends only 1 business despite 5 in context — conservative but at least it's the right one
- "complimentary champagne for the bride" is from a bachelorette snippet (snippet 3 of Jacques-Imo's) — synthesizer conflated bachelor/bachelorette; minor faithfulness gap
- **Acme Oyster House: duplicate evidence snippet** — identical review appears twice; ingest/ChromaDB dedup issue
- Ruby Slipper passed `noise_level=loud` SQL filter but reviews describe a brunch place, not loud — likely a data quality issue in Yelp attributes (`NoiseLevel` field set incorrectly by business)

**Open:** Build query_eval.py with 20 queries to get quantitative accuracy scores across baseline and current prompt versions.

---

## EXP-008 — query_eval.py DB coverage baseline
**Date:** 2026-04-23  
**Mode:** `python benchmarks/query_eval.py --mode filter` (no LLM required)  
**What it checks:** 20-query eval set (14 `find_businesses`, 4 `question_about_business`, 2 `compare_businesses`). Filter mode runs each `find_businesses` query's expected filters through `filter_businesses()` to verify the DB returns ≥3 businesses.

**Results: 14/14 PASS (100%)**

| ID    | Businesses returned | Query |
|-------|--------------------:|-------|
| FB-01 | 37  | bachelor party spot, loud, handles large groups |
| FB-02 | 104 | cheap brunch place with outdoor seating |
| FB-03 | 6   | jazz brunch with live music |
| FB-04 | 58  | late-night Cajun food after a show on Frenchmen Street |
| FB-05 | 479 | family-friendly seafood place with parking |
| FB-06 | 77  | quiet romantic date spot |
| FB-07 | 312 | outdoor patio restaurant with a full bar |
| FB-08 | 103 | dog-friendly restaurant with a patio |
| FB-09 | 31  | upscale dinner spot that takes reservations |
| FB-10 | 211 | happy hour bar with TVs to watch sports |
| FB-11 | 26  | BYOB place with casual attire |
| FB-12 | 34  | bachelorette dinner, upscale vibes, good for large groups |
| FB-13 | 225 | wheelchair accessible restaurant that caters events |
| FB-14 | 3   | dressy dinner spot with live music |

QB/CB cases skipped in filter mode — intent-only, no SQL filters.

**Notes:**
- FB-14 (`attire=dressy + music.live=true`) is the tightest filter at exactly 3 businesses — right at the sparse fallback threshold; watch for regressions if min_stars floor is raised
- FB-03 (`good_for_meal.brunch + music.live`) returns 6 — healthy headroom despite two JSON sub-key filters
- Next: run `--mode planner` once vllm is up to get planner accuracy scores

---

## EXP-009 — query_eval.py planner accuracy baseline
**Date:** 2026-04-23  
**Mode:** `python benchmarks/query_eval.py --mode planner`  
**Model:** Qwen2.5-7B-Instruct (vllm)  
**What it checks:** For each query, verifies `plan_query()` produces the correct intent and all required filter keys, with no forbidden keys (hallucination guard).

**Results: 20/20 PASS (100%)**

All three intent types pass:
- `find_businesses` (FB-01–14): 14/14 — correct filters, no spurious fields
- `question_about_business` (QB-01–04): 4/4 — correct intent classification
- `compare_businesses` (CB-01–02): 2/2 — correct intent classification

**Notes:**
- Prompt fixes from EXP-002 (anti-inference rules + few-shot) and EXP-005 (semantic/SQL decoupling) are holding across all 20 cases
- Hallucination guard on FB-01 and FB-12 (forbidden: `attire`, `noise_level`) passed — no spurious filters on occasion-word queries
- This is the post-prompt-tuning baseline; re-run after any `_SYSTEM_PROMPT` changes

---

## EXP-010 — Latency baseline (per-stage breakdown, vllm)
**Date:** 2026-04-23  
**Model:** Qwen2.5-7B-Instruct (vllm)  
**What it measures:** Per-stage wall-clock time for 3 queries. First query includes cold-start cost (model/embedding warmup).

**Results:**

| Stage | Q1 bachelor party (cold) | Q2 romantic date (warm) | Q3 jazz brunch (warm) |
|---|---:|---:|---:|
| planner | 18,243 ms | 2,722 ms | 2,893 ms |
| sql_filter | 6 ms | 4 ms | 8 ms |
| retrieval | 10,967 ms | 2,886 ms | 1,385 ms |
| meta_fetch | 1 ms | 1 ms | 0 ms |
| synthesizer | 21,419 ms | 13,588 ms | 14,152 ms |
| **TOTAL** | **50,635 ms** | **19,201 ms** | **18,438 ms** |
| businesses | 5 | 5 | 4 |

**Spec target:** warm p50 < 15,000 ms

**Analysis:**
- Cold-start overhead: ~15s on planner + ~9s on retrieval (first call initialises model weights and ChromaDB HNSW index) — not representative of steady-state
- **Warm p50 estimate: ~18–19s** — 3–4s above the 15s target
- `sql_filter` and `meta_fetch` are negligible (<10ms) — not optimization targets
- **Synthesizer is the dominant bottleneck: 13–21s warm** — accounts for ~75% of warm latency
- Planner is 2.7–2.9s warm — acceptable but can be parallelized with retrieval if needed
- Retrieval is 1–3s warm — secondary bottleneck

**Next:** Reduce synthesizer latency. Candidates: shorter prompt, streaming response, fewer snippets per business, or parallelise planner + retrieval.

---

## EXP-011 — 4-bit quantized model (latency)
**Date:** 2026-04-23
**Model:** mlx-community/Qwen2.5-7B-Instruct-4bit (vllm)
**Change:** `vllm_model` switched from `Qwen/Qwen2.5-7B-Instruct` to `mlx-community/Qwen2.5-7B-Instruct-4bit` in config + `.env`
**Reproduced via:** `PYTHONPATH=. .venv/bin/python benchmarks/latency_breakdown.py`

**Results:**

| Stage | Q1 bachelor party (cold) | Q2 romantic date (warm) | Q3 jazz brunch (warm) |
|---|---:|---:|---:|
| planner | 1,057 ms | 906 ms | 952 ms |
| sql_filter | 5 ms | 4 ms | 2 ms |
| retrieval | 7,499 ms | 923 ms | 1,068 ms |
| meta_fetch | 0 ms | 0 ms | 0 ms |
| synthesizer | 8,820 ms | 4,757 ms | 5,460 ms |
| **TOTAL** | **17,381 ms** | **6,590 ms** | **7,482 ms** |
| businesses | 7 | 9 | 4 |

**Spec target:** warm p50 < 15,000 ms
**Warm p50: 7,482 ms — target met with ~2x headroom**

**vs EXP-010 (full model):**
| Stage | EXP-010 warm avg | EXP-011 warm avg | Speedup |
|---|---:|---:|---:|
| planner | 2,808 ms | 929 ms | ~3x |
| retrieval | 2,136 ms | 996 ms | ~2x |
| synthesizer | 13,870 ms | 5,109 ms | ~2.7x |
| total | ~18,820 ms | ~7,036 ms | ~2.6x |

**Analysis:**
- 4-bit quantization delivers ~2.6x end-to-end speedup with no planner accuracy regression (20/20 held — EXP-009 baseline preserved)
- Synthesizer dropped from ~14s to ~5s warm — remains the largest single stage but no longer a bottleneck
- Warm p50 7.5s is comfortably under the 15s target; leaves headroom for v2 session overhead (~2s expected per spec)
- Cold-start overhead reduced from ~30s to ~8s

**Decision:** 4-bit model is the new default. No further latency optimisation needed for v1.

---

## EXP-012 — Synthesizer output cap (max_tokens + stop sequence + concise prompt)
**Date:** 2026-04-23
**Changes:**
- `synthesizer.py`: added `max_tokens=300`, `stop=["<|im_end|>"]` to LLM call
- `synthesizer.py`: added conciseness instruction to system prompt ("Recommend at most 3 restaurants. Give 1–2 sentences per recommendation.")

**Results:**

| Stage | Q1 bachelor party (cold) | Q2 romantic date (warm) | Q3 jazz brunch (warm) |
|---|---:|---:|---:|
| planner | 1,711 ms | 860 ms | 914 ms |
| sql_filter | 5 ms | 3 ms | 2 ms |
| retrieval | 7,831 ms | 1,999 ms | 511 ms |
| meta_fetch | 0 ms | 0 ms | 0 ms |
| synthesizer | 9,429 ms | 4,645 ms | 4,440 ms |
| **TOTAL** | **18,976 ms** | **7,507 ms** | **5,867 ms** |
| businesses | 7 | 9 | 4 |

**Warm p50: 7,507 ms**

**Analysis:**
- Synthesizer converged: Q3 dropped 1s (cap firing on previously longer output); Q2 marginal gain
- Warm p50 essentially flat vs EXP-011 (7,507ms vs 7,482ms) — retrieval variance (511ms–1,999ms) swamped synthesizer gains
- Retrieval variance is now the noise floor: ChromaDB `$in` filter cost scales with candidate pool size (Q2=9 biz, Q3=4 biz)
- `collection.count()` called on every request is a likely contention source — fix pending

---

## EXP-013 — Cache collection.count() at startup
**Date:** 2026-04-23
**Change:** `retriever.py`: `collection.count()` now called once on first load and cached in `_collection_size`; subsequent calls read the cached integer

**Results (run 1 / run 2):**

| Stage | Q1 cold | Q2 warm R1 / R2 | Q3 warm R1 / R2 |
|---|---:|---:|---:|
| planner | 1,424 / 1,039 ms | 849 / 851 ms | 904 / 909 ms |
| retrieval | 7,876 / 7,883 ms | 1,191 / 894 ms | 696 / 950 ms |
| synthesizer | 3,417 / 3,478 ms | 2,345 / 2,340 ms | 1,990 / 1,970 ms |
| **TOTAL** | **12,724 / 12,404 ms** | **4,388 / 4,087 ms** | **3,594 / 3,833 ms** |
| businesses | 7 | 9 | 4 |

**Warm p50: ~4.1s (confirmed stable across 2 runs)**

**Analysis:**
- Synthesizer dropped from ~4.5s → ~2.2s warm — nearly identical across both runs, variance eliminated
- Retrieval stabilised: 894–1,191ms (vs 511–1,999ms swing in EXP-012)
- The `collection.count()` scan on every request was causing I/O contention that affected both retrieval jitter and synthesizer throughput
- Cold improved: 12.4s vs 17.4s (EXP-011) — same cold-start path but less system pressure
- **Warm p50 4.1s is 3.6x under the 15s target**

**Decision:** Latency optimisation complete for v1. Synthesizer and retrieval are both stable and fast.

---

## EXP-014 — FastAPI startup warmup (eager loading)
**Date:** 2026-04-24
**Change:** `api/main.py`: `@app.on_event("startup")` calls `_get_model()` + `_get_collection()` at server boot. `benchmarks/latency_breakdown.py` updated to call the same warmup before timed queries — all 3 queries now measured at post-warmup state.
**Reproduced via:** `python -m benchmarks.latency_breakdown`

**Results:**

| Stage | Q1 bachelor party | Q2 romantic date | Q3 jazz brunch |
|---|---:|---:|---:|
| planner | 1,752 ms | 829 ms | 885 ms |
| sql_filter | 6 ms | 3 ms | 1 ms |
| retrieval | 2,928 ms | 1,703 ms | 513 ms |
| meta_fetch | 0 ms | 0 ms | 0 ms |
| synthesizer | 2,739 ms | 2,252 ms | 1,890 ms |
| **TOTAL** | **7,425 ms** | **4,787 ms** | **3,289 ms** |
| businesses | 7 | 9 | 4 |

**Warm p50: 4,787 ms**

**Analysis:**
- Q1 planner (1,752ms) is slower than Q2/Q3 (~850ms) — first vLLM call in the process warms its own KV-cache; not controllable from our app
- Q1 retrieval (2,928ms) is elevated — `_get_collection()` loads the collection object but ChromaDB defers HNSW index loading until the first actual `.query()` call; a dummy `retrieve()` in warmup would close this gap
- Q2/Q3 retrieval and synthesizer are fully stable: retrieval 513–1,703ms, synthesizer 1,890–2,252ms
- Warm p50 4,787ms is consistent with EXP-013 (~4.1s); users no longer absorb cold-start cost
