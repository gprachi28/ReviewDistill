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
