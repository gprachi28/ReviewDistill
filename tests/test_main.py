"""
Tests for api/main.py

Uses FastAPI TestClient — no vLLM or ChromaDB required.
"""
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from api.main import app
from api.schemas import AnalyzeResponse, SentimentScore

client = TestClient(app)

_SAMPLE_RESPONSE = AnalyzeResponse(
    sentiment=SentimentScore(positive=0.7, neutral=0.2, negative=0.1),
    review_count=10,
    latency_ms=500,
)


def test_v1_analyze_returns_200_with_sentiment():
    with patch("api.main.analyze_v1", return_value=_SAMPLE_RESPONSE):
        response = client.post("/api/v1/analyze", json={"business_id": "biz_001"})

    assert response.status_code == 200
    body = response.json()
    assert body["sentiment"]["positive"] == 0.7
    assert body["review_count"] == 10
    assert body["summary"] is None


def test_v1_analyze_returns_404_for_unknown_business():
    with patch("api.main.analyze_v1", side_effect=ValueError("No reviews found for business_id='biz_unknown'")):
        response = client.post("/api/v1/analyze", json={"business_id": "biz_unknown"})

    assert response.status_code == 404
    assert "biz_unknown" in response.json()["detail"]


def test_v1_analyze_returns_422_for_missing_business_id():
    response = client.post("/api/v1/analyze", json={})
    assert response.status_code == 422
