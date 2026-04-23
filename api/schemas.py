"""
api/schemas.py

Pydantic models for all request/response types across v1 and v2 endpoints.
"""
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field

Intent = Literal["find_businesses", "question_about_business", "compare_businesses"]


class QueryPlan(BaseModel):
    intent: Intent
    sql_filters: dict[str, Any]
    semantic_query: str


class BusinessResult(BaseModel):
    business_id: str
    name: str
    stars: float
    price_range: Optional[int]
    evidence: list[str]  # all snippets passed to the LLM, ordered by relevance


# ── v1 ─────────────────────────────────────────────────────────────────────────


class QueryRequest(BaseModel):
    question: str = Field(..., max_length=2000)


class QueryResponse(BaseModel):
    answer: str
    businesses: list[BusinessResult]
    query_plan: QueryPlan
    cache_hit: bool = False
    latency_ms: int


# ── v2 ─────────────────────────────────────────────────────────────────────────


class SessionQueryRequest(BaseModel):
    question: str
    session_id: Optional[str] = None


class SessionQueryResponse(QueryResponse):
    session_id: str
