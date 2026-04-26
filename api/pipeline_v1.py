"""
api/pipeline_v1.py

Single-shot query pipeline (v1): stateless, no session context.

Public API:
    run(question) -> QueryResponse
    stream(question) -> Iterator[str]  — SSE-formatted events
"""
import json
import sqlite3
import time
from collections.abc import Iterator

from api.query_planner import plan_query
from api.retriever import retrieve
from api.schemas import QueryResponse
from api.sql_filter import filter_businesses
from api.synthesizer import synthesize, synthesize_stream
from config import settings


def run(question: str) -> QueryResponse:
    """
    Execute the full query pipeline for a single stateless question.

    Steps:
        1. Query Planner  — question → {sql_filters, semantic_query, intent}
        2. SQL Filter     — sql_filters → candidate business_ids (or None for semantic-only)
        3. Retrieval      — semantic_query + candidate_ids → ranked review snippets
        4. Meta fetch     — business_ids from snippets → {name, stars, price_range}
        5. Synthesizer    — snippets + meta → conversational answer + BusinessResult list
    """
    t0 = time.time()

    query_plan = plan_query(question)

    candidate_ids = filter_businesses(query_plan.sql_filters)

    snippets = retrieve(query_plan.semantic_query, candidate_ids)

    biz_ids = list({s["business_id"] for s in snippets})
    business_meta = _fetch_business_meta(biz_ids)

    answer, businesses = synthesize(question, snippets, business_meta)

    latency_ms = int((time.time() - t0) * 1000)

    return QueryResponse(
        answer=answer,
        businesses=businesses,
        query_plan=query_plan,
        latency_ms=latency_ms,
    )


def stream(question: str) -> Iterator[str]:
    """
    Execute the pipeline and yield SSE-formatted strings at each stage boundary.

    Event sequence:
        planning   — after Query Planner; carries intent, sql_filters, semantic_query
        candidates — after retrieval + meta fetch; carries business count and names
        token      — one per LLM token during synthesis; carries {"text": str}
        done       — after synthesis; carries full BusinessResult list
        error      — on any exception; carries {"message": str}
    """
    def sse(event: str, data: dict) -> str:
        return f"event: {event}\ndata: {json.dumps(data)}\n\n"

    try:
        query_plan = plan_query(question)
        yield sse("planning", query_plan.model_dump())

        candidate_ids = filter_businesses(query_plan.sql_filters)
        snippets = retrieve(query_plan.semantic_query, candidate_ids)
        biz_ids = list({s["business_id"] for s in snippets})
        business_meta = _fetch_business_meta(biz_ids)

        yield sse("candidates", {
            "count": len(biz_ids),
            "businesses": [m["name"] for m in business_meta.values()],
        })

        token_iter, businesses = synthesize_stream(question, snippets, business_meta)
        for token in token_iter:
            yield sse("token", {"text": token})

        yield sse("done", {"businesses": [b.model_dump() for b in businesses]})

    except Exception as exc:
        yield sse("error", {"message": str(exc)})


def _fetch_business_meta(business_ids: list[str]) -> dict[str, dict]:
    """Fetch name, stars, price_range from SQLite for the given business_ids."""
    if not business_ids:
        return {}
    placeholders = ",".join("?" * len(business_ids))
    conn = sqlite3.connect(settings.sqlite_path)
    rows = conn.execute(
        f"SELECT business_id, name, stars, price_range FROM businesses"
        f" WHERE business_id IN ({placeholders})",
        business_ids,
    ).fetchall()
    conn.close()
    return {row[0]: {"name": row[1], "stars": row[2], "price_range": row[3]} for row in rows}
