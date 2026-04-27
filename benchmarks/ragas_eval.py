"""
benchmarks/ragas_eval.py

Faithfulness evaluation using RAGAS 0.1.x.

Runs 14 find_businesses queries through pipeline_v1.run(), collects
(question, answer, contexts) per query, then evaluates with RAGAS
faithfulness metric using an independent judge LLM.

RAGAS faithfulness checks: for each atomic claim in the answer, is it
supported by the retrieved review snippets passed to the synthesizer?
Score is 0–1 (higher = fewer hallucinated claims).

Judge model:  gemini-2.5-flash via Google's OpenAI-compatible endpoint
              (requires GEMINI_API_KEY in .env)
              RAGAS faithfulness requires structured JSON output (claim
              decomposition + per-claim verdicts). Local 7B models do not
              reliably follow the required schema. Gemini 2.5 Flash is the
              minimum reliable judge for this metric.
Generator:    mlx-community/Qwen2.5-7B-Instruct-4bit on port 8001
              (via settings.llm_base_url)

Rate limiting:
  Samples are evaluated one at a time (not batched) with INTER_SAMPLE_SLEEP
  seconds between calls to avoid flooding the Gemini API. Each sample
  triggers two LLM calls inside RAGAS (claim extraction + verdict). On
  Gemini paid Tier 1 (1000 RPM) this is well within limits; the sleep is
  a conservative guard for burst protection.

Persistence:
  benchmarks/ragas_samples.json  — Qwen pipeline outputs, saved after each
                                   query. Re-run skips Step 1 if file exists.
  benchmarks/ragas_results.json  — Final per-query faithfulness scores.
                                   Saved incrementally — safe to interrupt.

Run:
    Terminal 1: .venv/bin/mlx_lm.server --model mlx-community/Qwen2.5-7B-Instruct-4bit --port 8001
    Terminal 2: PYTHONPATH=. .venv/bin/uvicorn api.main:app --port 8000
    Terminal 3: PYTHONPATH=. .venv/bin/python benchmarks/ragas_eval.py

    To re-run judge only (reuse saved Qwen samples):
    PYTHONPATH=. .venv/bin/python benchmarks/ragas_eval.py --judge-only
"""
import argparse
import json
import math
import sqlite3
import sys
import time
from pathlib import Path

sys.path.insert(0, ".")

from datasets import Dataset
from langchain_openai import ChatOpenAI
from ragas import evaluate
from ragas.executor import RunConfig
from ragas.metrics import faithfulness

from api.pipeline_v1 import run
from benchmarks.query_eval import EVAL_CASES
from config import settings

# SQLite columns to include in the injected metadata context string.
# These are the structured attributes the synthesizer has access to via SQL
# metadata but that never appear in anonymous review text — injecting them
# gives RAGAS visibility into both data sources the synthesizer uses.
_META_COLS = [
    "name", "stars", "price_range", "noise_level", "alcohol", "attire",
    "good_for_groups", "outdoor_seating", "dogs_allowed", "byob", "corkage",
    "good_for_dancing", "happy_hour", "has_tv", "takes_reservations",
    "good_for_kids", "caters", "wheelchair_accessible",
]

JUDGE_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
JUDGE_MODEL = "gemini-2.5-pro"

# Seconds to wait between per-sample judge calls — guards against 429 bursts.
# Two LLM calls per sample (claim extraction + verdicts), so effective rate
# is 2 calls / (eval_time + INTER_SAMPLE_SLEEP). At 3s sleep this is well
# under Gemini Tier 1's 1000 RPM ceiling.
INTER_SAMPLE_SLEEP = 3

SAMPLES_PATH = Path("benchmarks/ragas_samples.json")
RESULTS_PATH = Path("benchmarks/ragas_results.json")

# QB/CB cases don't follow retrieval → synthesis; faithfulness is only
# meaningful for find_businesses queries.
FB_CASES = [c for c in EVAL_CASES if c.expected_intent == "find_businesses"]


def collect_samples() -> list[dict]:
    """Run each query through the pipeline and save results incrementally."""
    if SAMPLES_PATH.exists():
        existing = json.loads(SAMPLES_PATH.read_text())
        done_questions = {s["question"] for s in existing}
        print(f"  Resuming — {len(existing)} samples already saved.")
    else:
        existing = []
        done_questions = set()

    samples = list(existing)

    for case in FB_CASES:
        if case.question in done_questions:
            print(f"{case.id}  SKIP (already saved)")
            continue

        print(f"{case.id}  {case.question[:65]}...")
        try:
            response = run(case.question)
        except Exception as exc:
            print(f"  ERROR: {exc}")
            continue

        # Prefix each snippet with its business name so RAGAS can verify
        # claims like "Trenasse is great for watching sports" by matching
        # "[Trenasse] sitting at bar watching football game" — without the
        # prefix, snippets are anonymous and RAGAS can't attribute them.
        contexts = [
            f"[{biz.name}] {snippet}"
            for biz in response.businesses
            for snippet in biz.evidence
        ]

        if not contexts:
            print("  SKIP — no evidence returned")
            continue

        sample = {
            "question": case.question,
            "answer": response.answer,
            "contexts": contexts,
            "business_ids": [b.business_id for b in response.businesses],
        }
        samples.append(sample)
        SAMPLES_PATH.write_text(json.dumps(samples, indent=2))
        print(f"  ✓  {len(response.businesses)} businesses · {len(contexts)} snippets  [saved]")

    return samples


def _build_metadata_contexts(business_ids: list[str]) -> list[str]:
    """
    Return one metadata string per business, serialised as a flat key-value
    sentence. Appended to the RAGAS contexts list so the judge can verify
    claims sourced from SQL metadata (business names, boolean attributes like
    dogs_allowed/byob, scalar attributes like attire) that never appear in
    anonymous review text.

    Only non-null, non-zero, non-empty values are included to keep strings
    short. Boolean fields stored as 1/0 are rendered as true/false.
    """
    if not business_ids:
        return []
    placeholders = ",".join("?" * len(business_ids))
    cols_sql = ", ".join(_META_COLS)
    conn = sqlite3.connect(settings.sqlite_path)
    rows = conn.execute(
        f"SELECT {cols_sql} FROM businesses WHERE business_id IN ({placeholders})",
        business_ids,
    ).fetchall()
    conn.close()

    bool_cols = {
        "good_for_groups", "outdoor_seating", "dogs_allowed", "byob", "corkage",
        "good_for_dancing", "happy_hour", "has_tv", "takes_reservations",
        "good_for_kids", "caters", "wheelchair_accessible",
    }
    meta_strings = []
    for row in rows:
        parts = []
        for col, val in zip(_META_COLS, row):
            if val is None or val == "" or val == 0:
                continue
            if col in bool_cols:
                parts.append(f"{col}: true")
            else:
                parts.append(f"{col}: {val}")
        if parts:
            meta_strings.append("Business metadata — " + " | ".join(parts))
    return meta_strings


def _inject_metadata_contexts(samples: list[dict]) -> list[dict]:
    """
    Return a copy of samples with business metadata appended to each contexts
    list. Samples without business_ids (collected before this fix) are passed
    through unchanged — they will still be scored, just without metadata.
    """
    enriched = []
    for s in samples:
        business_ids = s.get("business_ids", [])
        meta_contexts = _build_metadata_contexts(business_ids)
        enriched.append({**s, "contexts": s["contexts"] + meta_contexts})
    return enriched


def make_judge_llm() -> ChatOpenAI:
    """Build the Gemini judge LLM with retry/backoff for 429s."""
    return ChatOpenAI(
        model=JUDGE_MODEL,
        base_url=JUDGE_BASE_URL,
        api_key=settings.gemini_api_key,
        temperature=0.0,
        # LangChain's built-in tenacity retry handles 429 RateLimitError
        # with exponential backoff automatically when max_retries > 0.
        max_retries=6,
    )


def run_judge(samples: list[dict]) -> None:
    """Evaluate faithfulness one sample at a time with sleep between calls."""
    # Resume from partial results if interrupted mid-run
    if RESULTS_PATH.exists():
        existing_results = json.loads(RESULTS_PATH.read_text())
        done_questions = {r["question"] for r in existing_results.get("results", [])}
        all_records = existing_results.get("results", [])
        print(f"  Resuming judge — {len(done_questions)} results already saved.")
    else:
        done_questions = set()
        all_records = []

    # Inject business metadata into contexts so RAGAS can verify claims
    # sourced from SQL metadata (names, dogs_allowed, byob, attire, etc.)
    # that never appear in anonymous review snippets.
    samples = _inject_metadata_contexts(samples)

    judge_llm = make_judge_llm()
    # One sample at a time — prevents RAGAS from batching all 14 into a
    # single async burst that exhausts the per-minute quota.
    run_config = RunConfig(timeout=120, max_retries=6, max_wait=60, max_workers=1)

    print("=" * 65)
    print(f"Step 2 — RAGAS faithfulness eval (judge: {JUDGE_MODEL})")
    print(f"         {len(samples)} samples · {INTER_SAMPLE_SLEEP}s sleep between calls")
    print("=" * 65)

    for i, sample in enumerate(samples, 1):
        q = sample["question"]
        if q in done_questions:
            print(f"  [{i:02d}/{len(samples)}] SKIP (already scored)  {q[:55]}")
            continue

        print(f"  [{i:02d}/{len(samples)}] Scoring…  {q[:55]}")
        try:
            result = evaluate(
                Dataset.from_list([sample]),
                metrics=[faithfulness],
                llm=judge_llm,
                run_config=run_config,
            )
            df = result.to_pandas()
            record = json.loads(df.to_json(orient="records"))[0]
        except Exception as exc:
            print(f"           ERROR: {exc}")
            record = {"question": q, "answer": sample["answer"], "faithfulness": None}

        score = record.get("faithfulness")
        if score is None or (isinstance(score, float) and math.isnan(score)):
            print(f"           → N/A (failed)")
        else:
            bar = "█" * int(score * 20)
            print(f"           → {score:.2f}  {bar}")

        all_records.append(record)
        # Save after every sample so a crash doesn't lose prior work
        RESULTS_PATH.write_text(
            json.dumps({"judge": JUDGE_MODEL, "results": all_records}, indent=2)
        )

        if i < len(samples):
            time.sleep(INTER_SAMPLE_SLEEP)

    # Final summary
    valid_scores = [
        r["faithfulness"] for r in all_records
        if r.get("faithfulness") is not None and not math.isnan(r["faithfulness"])
    ]
    mean = sum(valid_scores) / len(valid_scores) if valid_scores else float("nan")

    print("\nPer-query scores:")
    print("─" * 65)
    for r in all_records:
        score = r.get("faithfulness")
        if score is None or math.isnan(score):
            print(f"  {r['question'][:52]:<54}  N/A")
        else:
            bar = "█" * int(score * 20)
            print(f"  {r['question'][:52]:<54}  {score:.2f}  {bar}")
    print("─" * 65)
    print(f"  Mean faithfulness: {mean:.3f}  ({len(valid_scores)}/{len(all_records)} queries scored)")
    print(f"  Judge: {JUDGE_MODEL}")
    print(f"\nResults saved to {RESULTS_PATH}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--judge-only", action="store_true",
                        help="Skip Qwen collection, load samples from ragas_samples.json")
    args = parser.parse_args()

    if args.judge_only:
        if not SAMPLES_PATH.exists():
            print(f"ERROR: {SAMPLES_PATH} not found. Run without --judge-only first.")
            sys.exit(1)
        samples = json.loads(SAMPLES_PATH.read_text())
        print(f"Loaded {len(samples)} samples from {SAMPLES_PATH}")
    else:
        print("=" * 65)
        print("Step 1 — collecting pipeline responses (Qwen on port 8001)")
        print("=" * 65)
        samples = collect_samples()
        print(f"\n{len(samples)}/{len(FB_CASES)} queries collected.\n")

        if not samples:
            print("No samples. Is mlx_lm.server running on port 8001?")
            sys.exit(1)

    run_judge(samples)


if __name__ == "__main__":
    main()
