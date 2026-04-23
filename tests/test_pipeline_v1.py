"""
Tests for api/pipeline_v1.py

All external calls (query planner, SQL filter, retriever, synthesizer, SQLite)
are mocked — tests verify orchestration logic only.
"""
from unittest.mock import MagicMock, patch

import pytest

from api.pipeline_v1 import run, _fetch_business_meta
from api.schemas import BusinessResult, QueryPlan, QueryResponse


# ── shared mocks ───────────────────────────────────────────────────────────────


def _make_plan(**kwargs):
    return QueryPlan(
        intent=kwargs.get("intent", "find_businesses"),
        sql_filters=kwargs.get("sql_filters", {"noise_level": "loud"}),
        semantic_query=kwargs.get("semantic_query", "loud bachelor party"),
    )


def _make_snippet(business_id="biz_a", text="Great spot.", distance=0.1):
    return {"business_id": business_id, "text": text, "stars": 4.5, "distance": distance}


def _make_business_result(business_id="biz_a"):
    return BusinessResult(
        business_id=business_id,
        name="Bayou Jazz",
        stars=4.5,
        price_range=2,
        evidence="Great spot.",
    )


# ── happy path ─────────────────────────────────────────────────────────────────


def test_run_returns_query_response():
    plan = _make_plan()
    snippets = [_make_snippet()]
    biz_result = _make_business_result()

    with (
        patch("api.pipeline_v1.plan_query", return_value=plan),
        patch("api.pipeline_v1.filter_businesses", return_value=["biz_a"]),
        patch("api.pipeline_v1.retrieve", return_value=snippets),
        patch("api.pipeline_v1._fetch_business_meta", return_value={"biz_a": {"name": "Bayou Jazz", "stars": 4.5, "price_range": 2}}),
        patch("api.pipeline_v1.synthesize", return_value=("Great choice!", [biz_result])),
    ):
        result = run("bachelor party spot")

    assert isinstance(result, QueryResponse)
    assert result.answer == "Great choice!"
    assert len(result.businesses) == 1
    assert result.query_plan == plan
    assert result.latency_ms >= 0


def test_run_passes_sql_filters_to_filter_businesses():
    plan = _make_plan(sql_filters={"noise_level": "loud", "good_for_groups": True})

    with (
        patch("api.pipeline_v1.plan_query", return_value=plan),
        patch("api.pipeline_v1.filter_businesses", return_value=["biz_a"]) as mock_filter,
        patch("api.pipeline_v1.retrieve", return_value=[]),
        patch("api.pipeline_v1._fetch_business_meta", return_value={}),
        patch("api.pipeline_v1.synthesize", return_value=("No results.", [])),
    ):
        run("question")

    mock_filter.assert_called_once_with({"noise_level": "loud", "good_for_groups": True})


def test_run_passes_candidate_ids_to_retrieve():
    plan = _make_plan()
    with (
        patch("api.pipeline_v1.plan_query", return_value=plan),
        patch("api.pipeline_v1.filter_businesses", return_value=["biz_a", "biz_b"]),
        patch("api.pipeline_v1.retrieve", return_value=[]) as mock_retrieve,
        patch("api.pipeline_v1._fetch_business_meta", return_value={}),
        patch("api.pipeline_v1.synthesize", return_value=("answer", [])),
    ):
        run("question")

    mock_retrieve.assert_called_once_with("loud bachelor party", ["biz_a", "biz_b"])


def test_run_passes_none_to_retrieve_on_semantic_only_fallback():
    plan = _make_plan()
    with (
        patch("api.pipeline_v1.plan_query", return_value=plan),
        patch("api.pipeline_v1.filter_businesses", return_value=None),
        patch("api.pipeline_v1.retrieve", return_value=[]) as mock_retrieve,
        patch("api.pipeline_v1._fetch_business_meta", return_value={}),
        patch("api.pipeline_v1.synthesize", return_value=("answer", [])),
    ):
        run("question")

    mock_retrieve.assert_called_once_with("loud bachelor party", None)


def test_run_fetches_meta_only_for_retrieved_businesses():
    plan = _make_plan()
    snippets = [
        _make_snippet("biz_a"),
        _make_snippet("biz_b"),
        _make_snippet("biz_a"),  # duplicate — should only fetch once
    ]

    with (
        patch("api.pipeline_v1.plan_query", return_value=plan),
        patch("api.pipeline_v1.filter_businesses", return_value=["biz_a", "biz_b", "biz_c"]),
        patch("api.pipeline_v1.retrieve", return_value=snippets),
        patch("api.pipeline_v1._fetch_business_meta", return_value={}) as mock_meta,
        patch("api.pipeline_v1.synthesize", return_value=("answer", [])),
    ):
        run("question")

    fetched_ids = set(mock_meta.call_args[0][0])
    assert fetched_ids == {"biz_a", "biz_b"}  # biz_c had no snippets


def test_run_cache_hit_defaults_false():
    plan = _make_plan()
    with (
        patch("api.pipeline_v1.plan_query", return_value=plan),
        patch("api.pipeline_v1.filter_businesses", return_value=None),
        patch("api.pipeline_v1.retrieve", return_value=[]),
        patch("api.pipeline_v1._fetch_business_meta", return_value={}),
        patch("api.pipeline_v1.synthesize", return_value=("answer", [])),
    ):
        result = run("question")

    assert result.cache_hit is False


# ── _fetch_business_meta ───────────────────────────────────────────────────────


def test_fetch_business_meta_empty_list_returns_empty():
    assert _fetch_business_meta([]) == {}


def test_fetch_business_meta_queries_sqlite(tmp_path):
    import sqlite3
    db = str(tmp_path / "test.db")
    conn = sqlite3.connect(db)
    conn.execute("""CREATE TABLE businesses (
        business_id TEXT, name TEXT, stars REAL, price_range INTEGER,
        review_count INTEGER, noise_level TEXT, alcohol TEXT, attire TEXT,
        wifi TEXT, smoking TEXT, good_for_groups INTEGER, takes_reservations INTEGER,
        outdoor_seating INTEGER, good_for_kids INTEGER, good_for_dancing INTEGER,
        happy_hour INTEGER, has_tv INTEGER, caters INTEGER, wheelchair_accessible INTEGER,
        dogs_allowed INTEGER, byob INTEGER, corkage INTEGER,
        ambience TEXT, good_for_meal TEXT, music TEXT, parking TEXT,
        categories TEXT, latitude REAL, longitude REAL
    )""")
    conn.execute("INSERT INTO businesses (business_id, name, stars, price_range, categories) "
                 "VALUES ('biz_a', 'Bayou Jazz', 4.5, 2, 'Restaurants')")
    conn.commit()
    conn.close()

    with patch("api.pipeline_v1.settings") as mock_settings:
        mock_settings.sqlite_path = db
        result = _fetch_business_meta(["biz_a"])

    assert result == {"biz_a": {"name": "Bayou Jazz", "stars": 4.5, "price_range": 2}}
