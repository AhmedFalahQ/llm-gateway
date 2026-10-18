
"""Structured logging in CloudWatch Embedded Metric Format."""

from __future__ import annotations

import json
import os
import sys
import time

NAMESPACE = os.environ.get("METRICS_NAMESPACE", "LLMGateway")


def emit(
    *,
    key_id: str,
    model: str,
    cost_usd: float,
    latency_ms: float,
    input_tokens: int,
    output_tokens: int,
    cache_tier: str = "miss",
    status: int = 200,
    stream=None,
) -> dict:
    """Write one EMF record and return it (returned so tests can assert on it)."""
    record = {
        "_aws": {
            "Timestamp": int(time.time() * 1000),
            "CloudWatchMetrics": [
                {
                    "Namespace": NAMESPACE,
                    "Dimensions": [["key_id", "model"]],
                    "Metrics": [
                        {"Name": "CostUSD", "Unit": "None"},
                        {"Name": "LatencyMs", "Unit": "Milliseconds"},
                        {"Name": "InputTokens", "Unit": "Count"},
                        {"Name": "OutputTokens", "Unit": "Count"},
                        {"Name": "CacheHit", "Unit": "Count"},
                    ],
                }
            ],
        },
        "key_id": key_id,
        "model": model,
        "CostUSD": round(cost_usd, 6),
        "LatencyMs": round(latency_ms, 2),
        "InputTokens": int(input_tokens),
        "OutputTokens": int(output_tokens),
        "CacheHit": 0 if cache_tier == "miss" else 1,
        "cache_tier": cache_tier,
        "status": status,
    }
    print(json.dumps(record), file=stream or sys.stdout)
    return record