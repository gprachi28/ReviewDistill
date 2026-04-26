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
              reliably follow the required schema. Gemini 2.0 Flash is the
              minimum reliable judge for this metric.
Generator:    mlx-community/Qwen2.5-7B-Instruct-4bit on port 8001
              (via settings.llm_base_url)

Persistence:
  benchmarks/ragas_samples.json  — Qwen pipeline outputs, saved after each
                                   query. Re-run skips Step 1 if file exists.
  benchmarks/ragas_results.json  — Final per-query faithfulness scores.

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
import sys
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

JUDGE_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
JUDGE_MODEL = "gemini-3-flash-preview"

SAMPLES_PATH = Path("benchmarks/ragas_samples.json")
RESULTS_PATH = Path("benchmarks/ragas_results.json")

# QB/CB cases don't follow retrieval → synthesis; faithfulness is only
# meaningful for find_businesses queries.
FB_CASES = [c for c in EVAL_CASES if c.expected_intent == "find_businesses"]


def collect_samples() -> list[dict]:
    """Run each query through the pipeline and save results incrementally."""
    # Resume from existing file if partially complete
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

        contexts = [snippet for biz in response.businesses for snippet in biz.evidence]

        if not contexts:
            print("  SKIP — no evidence returned")
            continue

        sample = {
            "question": case.question,
            "answer": response.answer,
            "contexts": contexts,
        }
        samples.append(sample)
        SAMPLES_PATH.write_text(json.dumps(samples, indent=2))
        print(f"  ✓  {len(response.businesses)} businesses · {len(contexts)} snippets  [saved]")

    return samples


def run_judge(samples: list[dict]) -> None:
    """Run RAGAS faithfulness eval and save results."""
    dataset = Dataset.from_list(samples)

    judge_llm = ChatOpenAI(
        model=JUDGE_MODEL,
        base_url=JUDGE_BASE_URL,
        api_key=settings.gemini_api_key,
        temperature=0.0,
    )

    print("=" * 65)
    print(f"Step 2 — RAGAS faithfulness eval (judge: {JUDGE_MODEL})")
    print("=" * 65)

    run_config = RunConfig(timeout=300, max_retries=2, max_wait=60, max_workers=1)
    result = evaluate(dataset, metrics=[faithfulness], llm=judge_llm, run_config=run_config)

    df = result.to_pandas()

    # Save results — use pandas JSON export to handle numpy types (ndarray, float32, etc.)
    records = json.loads(df.to_json(orient="records"))
    RESULTS_PATH.write_text(json.dumps({"judge": JUDGE_MODEL, "results": records}, indent=2))
    print(f"\nResults saved to {RESULTS_PATH}")

    print("\nPer-query scores:")
    print("─" * 65)
    for _, row in df.iterrows():
        score = row.get("faithfulness", float("nan"))
        if math.isnan(score):
            print(f"  {row['question'][:52]:<54}  N/A  (failed)")
        else:
            bar = "█" * int(score * 20)
            print(f"  {row['question'][:52]:<54}  {score:.2f}  {bar}")

    valid = df["faithfulness"].dropna()
    mean = valid.mean()
    print("─" * 65)
    print(f"  Mean faithfulness: {mean:.3f}  ({len(valid)}/{len(df)} queries scored)")
    print(f"  Judge: {JUDGE_MODEL}")


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
