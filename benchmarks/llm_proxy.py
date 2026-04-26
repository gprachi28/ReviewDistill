"""
benchmarks/llm_proxy.py

Round-robin proxy across two mlx_lm.server instances.
Exposes an OpenAI-compatible endpoint on port 8003 and alternates
each request between backends on ports 8001 and 8002.

Setup:
    Terminal 1:  .venv/bin/mlx_lm.server --model mlx-community/Qwen2.5-7B-Instruct-4bit --port 8001
    Terminal 2:  .venv/bin/mlx_lm.server --model mlx-community/Qwen2.5-7B-Instruct-4bit --port 8002
    Terminal 3:  PYTHONPATH=. .venv/bin/uvicorn benchmarks.llm_proxy:app --port 8003
    .env:        LLM_BASE_URL=http://localhost:8003/v1

Then start the API and Locust as normal.
"""
import itertools
import logging

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import Response

logger = logging.getLogger(__name__)

BACKENDS = [
    "http://localhost:8001",
    "http://localhost:8002",
]

_cycle = itertools.cycle(BACKENDS)

app = FastAPI()


@app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"])
async def proxy(request: Request, path: str) -> Response:
    backend = next(_cycle)
    url = f"{backend}/{path}"
    logger.info("→ %s %s", request.method, url)

    headers = {k: v for k, v in request.headers.items() if k.lower() != "host"}
    body = await request.body()

    async with httpx.AsyncClient(timeout=600.0) as client:
        resp = await client.request(
            method=request.method,
            url=url,
            content=body,
            headers=headers,
            params=dict(request.query_params),
        )

    return Response(
        content=resp.content,
        status_code=resp.status_code,
        headers=dict(resp.headers),
    )
