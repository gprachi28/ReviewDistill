"""
Tests for api/synthesizer.py

Mocks the LLM call — tests prompt construction, evidence grouping, and response parsing.
"""
from unittest.mock import MagicMock, patch

import pytest

from api.synthesizer import synthesize
from api.schemas import BusinessResult


def _mock_llm(answer: str):
    response = MagicMock()
    response.choices[0].message.content = answer
    client = MagicMock()
    client.chat.completions.create.return_value = response
    return patch("api.synthesizer._client", client), client


# ── fixtures ───────────────────────────────────────────────────────────────────


SNIPPETS = [
    {"business_id": "biz_a", "text": "Loud and fun, great for groups.", "stars": 5.0, "distance": 0.1},
    {"business_id": "biz_a", "text": "Best jazz brunch in the city.", "stars": 4.0, "distance": 0.2},
    {"business_id": "biz_b", "text": "Intimate setting, perfect for a date.", "stars": 4.5, "distance": 0.3},
]

BIZ_META = {
    "biz_a": {"name": "Bayou Jazz", "stars": 4.5, "price_range": 2},
    "biz_b": {"name": "The Quiet Creole", "stars": 4.2, "price_range": 3},
}


# ── happy path ─────────────────────────────────────────────────────────────────


def test_synthesize_returns_answer_and_businesses():
    mock_patch, _ = _mock_llm("Bayou Jazz is perfect for a bachelor party.")
    with mock_patch:
        answer, businesses = synthesize("bachelor party spot", SNIPPETS, BIZ_META)

    assert answer == "Bayou Jazz is perfect for a bachelor party."
    assert len(businesses) == 2
    assert all(isinstance(b, BusinessResult) for b in businesses)


def test_synthesize_business_names_from_meta():
    mock_patch, _ = _mock_llm("Here are two spots.")
    with mock_patch:
        _, businesses = synthesize("any question", SNIPPETS, BIZ_META)

    names = {b.name for b in businesses}
    assert names == {"Bayou Jazz", "The Quiet Creole"}


def test_synthesize_evidence_is_top_snippet():
    mock_patch, _ = _mock_llm("Some answer.")
    with mock_patch:
        _, businesses = synthesize("question", SNIPPETS, BIZ_META)

    biz_a = next(b for b in businesses if b.business_id == "biz_a")
    # Top snippet for biz_a is the first one (lowest distance)
    assert biz_a.evidence == "Loud and fun, great for groups."


def test_synthesize_business_result_fields():
    mock_patch, _ = _mock_llm("Answer.")
    with mock_patch:
        _, businesses = synthesize("question", SNIPPETS, BIZ_META)

    biz_b = next(b for b in businesses if b.business_id == "biz_b")
    assert biz_b.stars == 4.2
    assert biz_b.price_range == 3


# ── prompt construction ────────────────────────────────────────────────────────


def test_synthesize_prompt_includes_question():
    mock_patch, client = _mock_llm("answer")
    with mock_patch:
        synthesize("jazz brunch bachelor party", SNIPPETS, BIZ_META)

    user_message = client.chat.completions.create.call_args[1]["messages"][1]["content"]
    assert "jazz brunch bachelor party" in user_message


def test_synthesize_prompt_includes_review_text():
    mock_patch, client = _mock_llm("answer")
    with mock_patch:
        synthesize("any question", SNIPPETS, BIZ_META)

    user_message = client.chat.completions.create.call_args[1]["messages"][1]["content"]
    assert "Loud and fun, great for groups." in user_message
    assert "Intimate setting" in user_message


def test_synthesize_prompt_caps_snippets_per_business():
    # 4 snippets for biz_a, snippets_per_business=2 → only first 2 in prompt
    many_snippets = [
        {"business_id": "biz_a", "text": f"Review {i}", "stars": 4.0, "distance": float(i)}
        for i in range(4)
    ]
    mock_patch, client = _mock_llm("answer")
    with mock_patch:
        synthesize("question", many_snippets, {"biz_a": {"name": "Bayou", "stars": 4.0, "price_range": 2}}, snippets_per_business=2)

    user_message = client.chat.completions.create.call_args[1]["messages"][1]["content"]
    assert "Review 0" in user_message
    assert "Review 1" in user_message
    assert "Review 2" not in user_message
    assert "Review 3" not in user_message


# ── edge cases ─────────────────────────────────────────────────────────────────


def test_synthesize_excludes_businesses_not_in_meta():
    # biz_c has a snippet but no metadata — should be excluded
    snippets_with_unknown = SNIPPETS + [
        {"business_id": "biz_c", "text": "Unknown place.", "stars": 3.0, "distance": 0.5}
    ]
    mock_patch, _ = _mock_llm("answer")
    with mock_patch:
        _, businesses = synthesize("question", snippets_with_unknown, BIZ_META)

    ids = {b.business_id for b in businesses}
    assert "biz_c" not in ids


def test_synthesize_empty_snippets_returns_fallback():
    mock_patch, client = _mock_llm("unused")
    with mock_patch:
        answer, businesses = synthesize("question", [], BIZ_META)

    assert "couldn't find" in answer.lower()
    assert businesses == []
    client.chat.completions.create.assert_not_called()
