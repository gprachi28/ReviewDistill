"""
benchmarks/load_test.py

Run:
    locust -f benchmarks/load_test.py --host http://localhost:8000 \
      --users 10 --spawn-rate 2 --run-time 60s --headless
"""
import random

from locust import HttpUser, between, task

BUSINESS_IDS = ["biz_001", "biz_002"]  # extend with real IDs from your dataset


class ReviewAnalysisUser(HttpUser):
    wait_time = between(1, 3)

    @task(3)
    def analyze_v1(self):
        self.client.post(
            "/api/v1/analyze",
            json={"business_id": random.choice(BUSINESS_IDS)},
            name="/api/v1/analyze",
        )

    @task(2)
    def analyze_v2(self):
        self.client.post(
            "/api/v2/analyze",
            json={"business_id": random.choice(BUSINESS_IDS)},
            name="/api/v2/analyze",
        )

    @task(1)
    def analyze_v3(self):
        self.client.post(
            "/api/v3/analyze",
            json={"business_id": random.choice(BUSINESS_IDS)},
            name="/api/v3/analyze",
        )
