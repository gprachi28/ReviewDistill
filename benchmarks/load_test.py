"""
benchmarks/load_test.py

Run:
    locust -f benchmarks/load_test.py --host http://localhost:8000 \
      --users 5 --spawn-rate 1 --run-time 300s --headless
"""
import random

from locust import HttpUser, between, task

TEST_QUESTIONS = [
    "loud spot for a bachelor party that handles large groups",
    "romantic dinner with outdoor seating under $50",
    "jazz brunch with live music",
    "late-night Cajun food after a show on Frenchmen Street",
    "family-friendly seafood restaurant with parking",
]


class ConversationalQueryUser(HttpUser):
    wait_time = between(1, 3)

    @task
    def query_v1(self):
        self.client.post(
            "/api/v1/query",
            json={"question": random.choice(TEST_QUESTIONS)},
            name="/api/v1/query",
        )
