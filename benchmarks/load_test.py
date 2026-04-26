"""
benchmarks/load_test.py

Measures HTTP round-trip latency (cold vs warm) and server-reported latency
(from latency_ms in the response body) across 5 concurrent users.

Run:
    locust -f benchmarks/load_test.py --host http://localhost:8000 \
      --users 5 --spawn-rate 1 --run-time 300s --headless

Stats output includes two sets of rows:
  /api/v1/query [cold]   — first request per user (embedding model + HNSW cold)
  /api/v1/query [warm]   — subsequent requests
  server [cold]          — server-reported latency_ms, cold
  server [warm]          — server-reported latency_ms, warm
"""
import random

from locust import HttpUser, between, task

TEST_QUESTIONS = [
    "loud spot for a bachelor party that handles large groups",
    "romantic dinner with outdoor seating under $50",
    "jazz brunch with live music",
    "late-night Cajun food after a show on Frenchmen Street",
    "family-friendly seafood restaurant with parking",
    "quiet romantic date spot",
    "outdoor patio restaurant with a full bar",
    "dog-friendly restaurant with a patio",
    "upscale dinner spot that takes reservations",
    "happy hour bar with TVs to watch sports",
]


class ConversationalQueryUser(HttpUser):
    wait_time = between(1, 3)

    def on_start(self) -> None:
        self._is_cold = True

    @task
    def query_v1(self) -> None:
        label = "cold" if self._is_cold else "warm"

        with self.client.post(
            "/api/v1/query",
            json={"question": random.choice(TEST_QUESTIONS)},
            name=f"/api/v1/query [{label}]",
            catch_response=True,
        ) as resp:
            if resp.status_code != 200:
                resp.failure(f"HTTP {resp.status_code}")
                self._is_cold = False
                return

            try:
                server_ms = resp.json().get("latency_ms")
            except Exception:
                resp.failure("Invalid JSON response")
                self._is_cold = False
                return

            resp.success()

            if server_ms is not None:
                self.environment.events.request.fire(
                    request_type="server_latency",
                    name=f"server [{label}]",
                    response_time=server_ms,
                    response_length=0,
                    exception=None,
                    context={},
                )

        self._is_cold = False
