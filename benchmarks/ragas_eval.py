"""
benchmarks/ragas_eval.py

Run RAGAS evaluation on a sample of businesses.

Usage:
    python -m benchmarks.ragas_eval --version v1 --n_businesses 50
"""
import argparse
import json
import os

import httpx
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import answer_relevancy, faithfulness

API_BASE = "http://localhost:8000"


def load_sample_businesses(fixture_path: str, n: int) -> list[str]:
    data = json.load(open(fixture_path))
    ids = list({r["business_id"] for r in data})
    return ids[:n]


def query_api(version: str, business_id: str) -> dict:
    response = httpx.post(
        f"{API_BASE}/api/{version}/analyze",
        json={"business_id": business_id},
        timeout=60,
    )
    response.raise_for_status()
    return response.json()


def build_ragas_dataset(version: str, business_ids: list[str]) -> Dataset:
    questions, answers, contexts = [], [], []

    for biz_id in business_ids:
        try:
            result = query_api(version, biz_id)
        except Exception as e:
            print(f"Skipping {biz_id}: {e}")
            continue

        questions.append(f"Summarize reviews for business {biz_id}")
        answers.append(result.get("summary") or str(result["sentiment"]))
        contexts.append([f"business_id={biz_id}"])  # placeholder — extend with actual retrieved docs

    return Dataset.from_dict({
        "question": questions,
        "answer": answers,
        "contexts": contexts,
    })


def run(version: str, n_businesses: int, fixture_path: str) -> None:
    business_ids = load_sample_businesses(fixture_path, n_businesses)
    dataset = build_ragas_dataset(version, business_ids)

    results = evaluate(dataset, metrics=[faithfulness, answer_relevancy])
    print(f"\n=== RAGAS Results ({version}) ===")
    print(results)
    results.to_pandas().to_csv(f"benchmarks/ragas_{version}.csv", index=False)
    print(f"Saved to benchmarks/ragas_{version}.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", default="v1", choices=["v1", "v2", "v3"])
    parser.add_argument("--n_businesses", type=int, default=50)
    parser.add_argument("--fixture_path", default="fixtures/sample_reviews.json")
    args = parser.parse_args()
    run(args.version, args.n_businesses, args.fixture_path)
