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
