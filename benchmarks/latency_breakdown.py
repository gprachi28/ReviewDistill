"""
benchmarks/latency_breakdown.py

Per-stage latency breakdown for the v1 pipeline.

Run:
    PYTHONPATH=. .venv/bin/python benchmarks/latency_breakdown.py

Calls _get_model() and _get_collection() before timed queries to simulate
the warmed state produced by FastAPI's startup handler. All 3 queries are
measured at warm-state latency.

Matches the measurement methodology from EXP-010 onwards.
"""
import time

from api.query_planner import plan_query
from api.retriever import _get_collection, _get_model, retrieve
from api.sql_filter import filter_businesses
from api.synthesizer import synthesize
from api.pipeline_v1 import _fetch_business_meta

QUERIES = [
    "bachelor party spot, loud, handles large groups",
    "quiet romantic date spot",
    "jazz brunch with live music",
]


def run_with_breakdown(question: str) -> dict:
    stages = {}

    t = time.time()
    plan = plan_query(question)
    stages["planner"] = int((time.time() - t) * 1000)

    t = time.time()
    candidate_ids = filter_businesses(plan.sql_filters)
    stages["sql_filter"] = int((time.time() - t) * 1000)

    t = time.time()
    snippets = retrieve(plan.semantic_query, candidate_ids)
    stages["retrieval"] = int((time.time() - t) * 1000)

    t = time.time()
    biz_ids = list({s["business_id"] for s in snippets})
    business_meta = _fetch_business_meta(biz_ids)
    stages["meta_fetch"] = int((time.time() - t) * 1000)

    t = time.time()
    synthesize(question, snippets, business_meta)
    stages["synthesizer"] = int((time.time() - t) * 1000)

    stages["total"] = sum(stages.values())
    stages["businesses"] = len(biz_ids)
    return stages


def main():
    print("Warming up embedding model and ChromaDB index...")
    _get_model()
    _get_collection()
    retrieve("warmup")  # forces HNSW index into memory before timed queries
    print("Warmup complete.\n")

    header = f"{'Stage':<12}" + "".join(f"  Q{i+1:<18}" for i in range(len(QUERIES)))
    print(header)
    print("-" * len(header))

    all_results = []
    for i, q in enumerate(QUERIES):
        print(f"\nRunning Q{i+1}: {q!r}")
        all_results.append(run_with_breakdown(q))

    print()
    stages = ["planner", "sql_filter", "retrieval", "meta_fetch", "synthesizer", "total", "businesses"]
    for stage in stages:
        row = f"{stage:<12}"
        for r in all_results:
            row += f"  {r[stage]:>8} ms          " if stage != "businesses" else f"  {r[stage]:>8}              "
        print(row)

    warm = [r["total"] for r in all_results]
    print(f"\nWarm p50 estimate: {sorted(warm)[len(warm)//2]:,} ms  (target: <15,000 ms)")


if __name__ == "__main__":
    main()
