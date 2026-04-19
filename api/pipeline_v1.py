"""
api/pipeline_v1.py

Simple RAG: retrieve top-50 reviews → single LLM call → sentiment breakdown.
"""
import json
import time

from openai import OpenAI

from api.retriever import retrieve_reviews
from api.schemas import AnalyzeResponse, SentimentScore
from config import settings

_client = OpenAI(base_url=settings.vllm_base_url, api_key="not-needed")

_PROMPT = """You are analyzing customer reviews for a business.
Given the following reviews, provide a sentiment breakdown.

Reviews:
{reviews}

Respond in JSON: {{"positive": <float>, "neutral": <float>, "negative": <float>}}
Values must sum to 1.0."""


def analyze_v1(business_id: str) -> AnalyzeResponse:
    start = time.monotonic()
    reviews = retrieve_reviews(business_id, n_results=50)
    if not reviews:
        raise ValueError(f"No reviews found for business_id={business_id!r}")

    review_text = "\n---\n".join(r["text"] for r in reviews)
    response = _client.chat.completions.create(
        model=settings.vllm_model,
        messages=[{"role": "user", "content": _PROMPT.format(reviews=review_text)}],
        response_format={"type": "json_object"},
        temperature=0,
    )
    data = json.loads(response.choices[0].message.content)
    latency_ms = int((time.monotonic() - start) * 1000)

    return AnalyzeResponse(
        sentiment=SentimentScore(**data),
        review_count=len(reviews),
        latency_ms=latency_ms,
    )
