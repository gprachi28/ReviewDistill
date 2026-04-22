"""
benchmarks/load_test.py

Run:
    locust -f benchmarks/load_test.py --host http://localhost:8000 \
      --users 5 --spawn-rate 1 --run-time 300s --headless
"""
import random

from locust import HttpUser, between, task

BUSINESS_IDS = ["biz_001", "biz_002"]  # extend with real IDs from your dataset


class ReviewAnalysisUser(HttpUser):
    wait_time = between(1, 3)

    @task
    def analyze_v1(self):
        self.client.post(
            "/api/v1/analyze",
            json={"business_id": random.choice(BUSINESS_IDS)},
            name="/api/v1/analyze",
        )
