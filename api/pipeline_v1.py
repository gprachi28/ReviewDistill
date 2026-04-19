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

_PROMPT = """You are a sentiment analyst evaluating customer reviews for a business.

You will be given {n_reviews} customer reviews, each prefixed with its star rating (1-5).
Analyse them together and return a sentiment breakdown as a JSON object.

Scoring guide:
- 4-5 stars -> positive
- 3 stars -> use the review text to decide; do not default to neutral
- 1-2 stars -> negative

Reviews:
{reviews}

Return only valid JSON with exactly these keys:
{{"positive": <float 0-1>, "neutral": <float 0-1>, "negative": <float 0-1>}}
The three values must sum to 1.0. No explanation, no markdown, no extra keys."""


def analyze_v1(business_id: str) -> AnalyzeResponse:
    start = time.monotonic()
    reviews = retrieve_reviews(business_id, n_results=50)
    if not reviews:
        raise ValueError(f"No reviews found for business_id={business_id!r}")

    review_text = "\n---\n".join(
        f"[{r['stars']} stars] {r['text']}" for r in reviews
    )
    response = _client.chat.completions.create(
        model=settings.vllm_model,
        messages=[{"role": "user", "content": _PROMPT.format(n_reviews=len(reviews), reviews=review_text)}],
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
