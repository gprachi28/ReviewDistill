"""
Tests for api/schemas.py
"""
import pytest
from pydantic import ValidationError

from api.schemas import (
    BusinessResult,
    QueryPlan,
    QueryRequest,
    QueryResponse,
    SessionQueryRequest,
    SessionQueryResponse,
)


# ── QueryPlan ──────────────────────────────────────────────────────────────────


def test_query_plan_valid():
    plan = QueryPlan(
        intent="find_businesses",
        sql_filters={"noise_level": ["loud", "very_loud"], "good_for_groups": True},
        semantic_query="loud bachelor party large groups",
    )
    assert plan.intent == "find_businesses"
    assert plan.sql_filters["good_for_groups"] is True


def test_query_plan_invalid_intent():
    with pytest.raises(ValidationError):
        QueryPlan(intent="unknown", sql_filters={}, semantic_query="test")


def test_query_plan_all_intents():
    for intent in ("find_businesses", "question_about_business", "compare_businesses"):
        plan = QueryPlan(intent=intent, sql_filters={}, semantic_query="x")
        assert plan.intent == intent


# ── BusinessResult ─────────────────────────────────────────────────────────────


def test_business_result_with_price_range():
    biz = BusinessResult(
        business_id="abc123",
        name="Bayou Spot",
        stars=4.3,
        price_range=2,
        evidence=["Great atmosphere for groups."],
    )
    assert biz.price_range == 2


def test_business_result_without_price_range():
    biz = BusinessResult(
        business_id="abc123",
        name="Bayou Spot",
        stars=4.3,
        price_range=None,
        evidence=["Great atmosphere."],
    )
    assert biz.price_range is None


# ── QueryRequest ───────────────────────────────────────────────────────────────


def test_query_request_valid():
    req = QueryRequest(question="Where should I eat for a jazz brunch?")
    assert req.question == "Where should I eat for a jazz brunch?"


def test_query_request_missing_question():
    with pytest.raises(ValidationError):
        QueryRequest()


# ── QueryResponse ──────────────────────────────────────────────────────────────


def _make_response(**kwargs):
    defaults = dict(
        answer="Here are some spots...",
        businesses=[],
        query_plan=QueryPlan(intent="find_businesses", sql_filters={}, semantic_query="x"),
        latency_ms=1200,
    )
    return QueryResponse(**{**defaults, **kwargs})


def test_query_response_defaults_cache_hit_false():
    resp = _make_response()
    assert resp.cache_hit is False


def test_query_response_cache_hit_true():
    resp = _make_response(cache_hit=True)
    assert resp.cache_hit is True


def test_query_response_with_businesses():
    biz = BusinessResult(business_id="x", name="Y", stars=4.0, price_range=1, evidence=["nice"])
    resp = _make_response(businesses=[biz])
    assert len(resp.businesses) == 1
    assert resp.businesses[0].name == "Y"


# ── SessionQueryRequest ────────────────────────────────────────────────────────


def test_session_request_without_session_id():
    req = SessionQueryRequest(question="Any late-night spots?")
    assert req.session_id is None


def test_session_request_with_session_id():
    req = SessionQueryRequest(question="What about parking?", session_id="abc123")
    assert req.session_id == "abc123"


# ── SessionQueryResponse ───────────────────────────────────────────────────────


def test_session_response_includes_session_id():
    resp = SessionQueryResponse(
        answer="Here are some spots...",
        businesses=[],
        query_plan=QueryPlan(intent="find_businesses", sql_filters={}, semantic_query="x"),
        latency_ms=800,
        session_id="sess-xyz",
    )
    assert resp.session_id == "sess-xyz"
    assert resp.cache_hit is False  # inherited default
