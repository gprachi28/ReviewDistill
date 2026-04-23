"""
Tests for api/query_planner.py

Mocks the LLM — tests schema parsing, retry logic, and history injection.
Does NOT make real LLM calls.
"""
import json
from unittest.mock import MagicMock, patch

import pytest

from api.query_planner import plan_query
from api.schemas import QueryPlan


def _make_raw(intent="find_businesses", sql_filters=None, semantic_query="test query"):
    return json.dumps({
        "intent": intent,
        "sql_filters": sql_filters or {},
        "semantic_query": semantic_query,
    })


def _mock_llm(return_values: list[str]):
    """Patch _call_llm to return successive values."""
    return patch("api.query_planner._call_llm", side_effect=return_values)


# ── happy path ─────────────────────────────────────────────────────────────────


def test_plan_query_returns_query_plan():
    raw = _make_raw(
        intent="find_businesses",
        sql_filters={"noise_level": ["loud", "very_loud"], "good_for_groups": True},
        semantic_query="loud bachelor party large groups",
    )
    with _mock_llm([raw]):
        result = plan_query("bachelor party spot, loud, handles large groups")

    assert isinstance(result, QueryPlan)
    assert result.intent == "find_businesses"
    assert result.sql_filters["good_for_groups"] is True
    assert result.semantic_query == "loud bachelor party large groups"


def test_plan_query_question_about_business():
    raw = _make_raw(intent="question_about_business", semantic_query="parking options Carbone")
    with _mock_llm([raw]):
        result = plan_query("What do people say about parking at Carbone?")
    assert result.intent == "question_about_business"


def test_plan_query_compare_businesses():
    raw = _make_raw(intent="compare_businesses", semantic_query="date night romantic atmosphere")
    with _mock_llm([raw]):
        result = plan_query("How does Dooky Chase compare to Galatoire's for a date night?")
    assert result.intent == "compare_businesses"


# ── markdown stripping ─────────────────────────────────────────────────────────


def test_plan_query_strips_markdown_fences():
    raw = "```json\n" + _make_raw() + "\n```"
    with _mock_llm([raw]):
        result = plan_query("any brunch spot?")
    assert isinstance(result, QueryPlan)


# ── retry on bad json ──────────────────────────────────────────────────────────


def test_plan_query_retries_on_bad_json():
    good = _make_raw(semantic_query="jazz brunch spot")
    with _mock_llm(["not json at all", good]):
        result = plan_query("jazz brunch spot")
    assert result.semantic_query == "jazz brunch spot"


def test_plan_query_retries_on_invalid_schema():
    # Missing required fields
    bad = json.dumps({"foo": "bar"})
    good = _make_raw(semantic_query="outdoor seating")
    with _mock_llm([bad, good]):
        result = plan_query("place with outdoor seating")
    assert result.semantic_query == "outdoor seating"


def test_plan_query_raises_after_two_failures():
    with _mock_llm(["bad json", "still bad json"]):
        with pytest.raises(ValueError, match="malformed output after retry"):
            plan_query("find me a restaurant")


# ── history injection ──────────────────────────────────────────────────────────


def test_plan_query_passes_history_to_llm():
    raw = _make_raw(semantic_query="parking options first restaurant")
    history = [
        {"role": "user", "content": "Find a jazz brunch spot"},
        {"role": "assistant", "content": "Here are three options..."},
    ]
    with patch("api.query_planner._call_llm", return_value=raw) as mock_call:
        plan_query("What about parking at the first one?", history=history)

    messages = mock_call.call_args[0][0]
    roles = [m["role"] for m in messages]
    # system + 2 history turns + current question
    assert roles == ["system", "user", "assistant", "user"]


def test_plan_query_caps_history_at_six_messages():
    raw = _make_raw()
    # 5 turns = 10 messages — should be capped to last 6
    history = [
        {"role": "user" if i % 2 == 0 else "assistant", "content": f"msg {i}"}
        for i in range(10)
    ]
    with patch("api.query_planner._call_llm", return_value=raw) as mock_call:
        plan_query("follow-up question", history=history)

    messages = mock_call.call_args[0][0]
    # system (1) + 6 history + current question (1) = 8
    assert len(messages) == 8
