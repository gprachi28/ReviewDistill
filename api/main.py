"""
api/main.py

FastAPI application. Run with:
    uvicorn api.main:app --host 0.0.0.0 --port 8000
"""
from fastapi import FastAPI, HTTPException

from api.pipeline_v1 import run
from api.schemas import QueryRequest, QueryResponse

app = FastAPI(title="Yelp NOLA Conversational Assistant")


@app.post("/api/v1/query", response_model=QueryResponse)
def query_v1(request: QueryRequest) -> QueryResponse:
    try:
        return run(request.question)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
