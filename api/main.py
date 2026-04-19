from fastapi import FastAPI, HTTPException

from api.pipeline_v1 import analyze_v1
from api.schemas import AnalyzeRequest, AnalyzeResponse

app = FastAPI(title="yelp-rag-summarizer")


@app.post("/api/v1/analyze", response_model=AnalyzeResponse)
async def v1_analyze(request: AnalyzeRequest):
    try:
        return analyze_v1(request.business_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
