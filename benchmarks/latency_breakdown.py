"""
benchmarks/latency_breakdown.py

Per-stage latency breakdown for the v1 pipeline.

Run:
    python benchmarks/latency_breakdown.py

Runs 3 queries in order. First query includes cold-start overhead (model/ChromaDB
warmup) — reported separately. Warm queries are Q2 and Q3.

Matches the measurement methodology from EXP-010.
"""
import time

from api.query_planner import plan_query
from api.retriever import retrieve
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
    header = f"{'Stage':<12}" + "".join(f"  Q{i+1:<18}" for i in range(len(QUERIES)))
    print(header)
    print("-" * len(header))

    all_results = []
    for i, q in enumerate(QUERIES):
        label = "(cold)" if i == 0 else "(warm)"
        print(f"\nRunning Q{i+1} {label}: {q!r}")
        all_results.append(run_with_breakdown(q))

    print()
    stages = ["planner", "sql_filter", "retrieval", "meta_fetch", "synthesizer", "total", "businesses"]
    for stage in stages:
        row = f"{stage:<12}"
        for r in all_results:
            row += f"  {r[stage]:>8} ms          " if stage != "businesses" else f"  {r[stage]:>8}              "
        print(row)

    warm = [r["total"] for r in all_results[1:]]
    print(f"\nWarm p50 estimate: {sorted(warm)[len(warm)//2]:,} ms  (target: <15,000 ms)")


if __name__ == "__main__":
    main()
