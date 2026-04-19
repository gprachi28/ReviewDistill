"""
Tests for api/pipeline_v1.py

All LLM and retriever calls are mocked — no vLLM or ChromaDB required.
"""
from unittest.mock import MagicMock, patch

import pytest

from api.pipeline_v1 import analyze_v1
from api.schemas import AnalyzeResponse


def _mock_llm(content: str) -> MagicMock:
    response = MagicMock()
    response.choices[0].message.content = content
    return response


def test_analyze_v1_returns_analyze_response():
    reviews = [{"text": "Great food!", "stars": 5.0}]
    llm_json = '{"positive": 0.8, "neutral": 0.1, "negative": 0.1}'

    with patch("api.pipeline_v1.retrieve_reviews", return_value=reviews), \
         patch("api.pipeline_v1._client") as mock_client:
        mock_client.chat.completions.create.return_value = _mock_llm(llm_json)
        result = analyze_v1("biz_001")

    assert isinstance(result, AnalyzeResponse)
    assert result.summary is None
    assert result.sentiment.positive == 0.8
    assert result.sentiment.neutral == 0.1
    assert result.sentiment.negative == 0.1
    assert result.review_count == 1


def test_analyze_v1_raises_on_empty_reviews():
    with patch("api.pipeline_v1.retrieve_reviews", return_value=[]):
        with pytest.raises(ValueError, match="biz_unknown"):
            analyze_v1("biz_unknown")


def test_analyze_v1_sends_review_text_in_prompt():
    reviews = [
        {"text": "Excellent ramen.", "stars": 5.0},
        {"text": "Cold soup.", "stars": 1.0},
    ]
    llm_json = '{"positive": 0.5, "neutral": 0.0, "negative": 0.5}'

    with patch("api.pipeline_v1.retrieve_reviews", return_value=reviews), \
         patch("api.pipeline_v1._client") as mock_client:
        mock_client.chat.completions.create.return_value = _mock_llm(llm_json)
        analyze_v1("biz_002")

    call_args = mock_client.chat.completions.create.call_args
    prompt = call_args.kwargs["messages"][0]["content"]
    assert "Excellent ramen." in prompt
    assert "Cold soup." in prompt
