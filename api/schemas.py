from pydantic import BaseModel


class AnalyzeRequest(BaseModel):
    business_id: str


class SentimentScore(BaseModel):
    positive: float
    neutral: float
    negative: float


class AnalyzeResponse(BaseModel):
    summary: str | None = None
    sentiment: SentimentScore
    review_count: int
    latency_ms: int
