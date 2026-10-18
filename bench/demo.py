#!/usr/bin/env python3
"""End-to-end walkthrough of the gateway — offline, in a couple of seconds.

Drives the real Lambda handler with real API Gateway-shaped events and the
`echo` provider, so a reviewer can watch budgets, cache tiers and accounting
work without an AWS account and without spending anything.

Two backends, chosen by one environment variable:

    python bench/demo.py
        moto, in process. Nothing to start, nothing to clean up.

    AWS_ENDPOINT_URL_DYNAMODB=http://localhost:8000 python bench/demo.py
        Real DynamoDB Local over HTTP. Same transcript, real wire protocol,
        real conditional-update semantics. Tables are uniquely named and
        dropped afterwards.
"""

from __future__ import annotations

import contextlib
import io
import json
import os
import pathlib
import sys
import uuid

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from src import handler as handler_module  # noqa: E402
from src.budget import BudgetStore  # noqa: E402
from src.cache import SemanticCache  # noqa: E402
from src.providers.echo import EchoProvider  # noqa: E402
from src.tables import create_table  # noqa: E402

LOCAL_ENDPOINT = os.environ.get("AWS_ENDPOINT_URL_DYNAMODB")


@contextlib.contextmanager
def dynamodb_backend():
    """Yield (resource, label) for whichever backend is configured."""
    import boto3

    if LOCAL_ENDPOINT:
        # Credentials pinned, never inherited: the demo must not be able to
        # reach a real account because a profile is exported in the shell.
        yield (
            boto3.Session(
                aws_access_key_id="local",
                aws_secret_access_key="local",
                aws_session_token=None,
                region_name="us-east-1",
            ).resource("dynamodb"),
            f"DynamoDB Local at {LOCAL_ENDPOINT}",
        )
        return

    from moto import mock_aws

    with mock_aws():
        yield boto3.resource("dynamodb", region_name="us-east-1"), "moto (in process)"


@contextlib.contextmanager
def demo_tables(dynamodb):
    """Two uniquely named tables, dropped afterwards on the real backend."""
    suffix = uuid.uuid4().hex[:8]
    created = [
        create_table(dynamodb, f"demo-budgets-{suffix}"),
        create_table(dynamodb, f"demo-cache-{suffix}"),
    ]
    try:
        yield created
    finally:
        if LOCAL_ENDPOINT:
            for table in created:
                with contextlib.suppress(Exception):
                    table.delete()

API_KEY = "demo-key-abc123"
MODEL = "anthropic.claude-3-5-haiku-20241022-v1:0"

# Deliberately unrelated subjects. The echo provider embeds bag-of-words, so
# near-identical filler ("question 1", "question 2") would land in one bucket
# and be served from cache — which would make this step demonstrate the cache
# instead of the budget.
FILLER = [
    "Explain photosynthesis in two sentences.",
    "What causes tides?",
    "Summarize the rules of chess.",
    "How does a refrigerator work?",
    "Who composed the Brandenburg concertos?",
    "What is the tallest mountain in Africa?",
    "Describe how bread rises.",
    "Why is the sky blue?",
    "What does a central bank do?",
    "How do noise-cancelling headphones work?",
    "What is the capital of Peru?",
    "Explain plate tectonics briefly.",
    "How are pearls formed?",
    "What is the Doppler effect?",
    "Describe the water cycle.",
    "Why do leaves change colour?",
    "What is a black hole?",
    "How does Wi-Fi transmit data?",
    "What do bees do in winter?",
    "Explain compound interest.",
    "How is glass made?",
    "What is the speed of sound?",
    "Describe the structure of DNA.",
    "Why do cats purr?",
    "What is machine translation?",
    "How does a sundial work?",
    "What is the Sahara's area?",
    "Explain supply and demand.",
    "How do volcanoes erupt?",
    "What is sonar used for?",
]

# Every EMF metric line the handler prints is collected here instead of being
# interleaved with the transcript.
metrics: list[dict] = []


def event(method: str, path: str, body: dict | None = None) -> dict:
    return {
        "httpMethod": method,
        "path": path,
        "headers": {"Authorization": f"Bearer {API_KEY}"},
        "body": json.dumps(body) if body is not None else None,
    }


def invoke(ev: dict) -> tuple[int, dict]:
    """Call the handler, capturing its structured metric output."""
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        result = handler_module.handler(ev)
    for line in buffer.getvalue().splitlines():
        if line.strip():
            with contextlib.suppress(json.JSONDecodeError):
                metrics.append(json.loads(line))
    return result["statusCode"], json.loads(result["body"])


def call(label: str, ev: dict) -> dict:
    status, body = invoke(ev)
    print(f"\n--- {label}")
    print(f"    HTTP {status}")
    for field in ("text", "cached", "cache_tier", "similarity", "cost_usd", "saved_usd", "error"):
        if field in body:
            value = body[field]
            if isinstance(value, str) and len(value) > 64:
                value = value[:61] + "..."
            print(f"    {field:<12} {value}")
    if "budget" in body:
        b = body["budget"]
        print(f"    {'budget':<12} ${b['spend_usd']:.6f} spent / ${b['limit_usd']:.3f} limit")
    return body


def chat(content: str, **kw) -> dict:
    return {"model": MODEL, "messages": [{"role": "user", "content": content}], **kw}


REFUND_Q = "What is the refund window for online orders?"


def main() -> int:
    try:
        import boto3  # noqa: F401 - imported for the error message below
    except ImportError:
        print("this demo needs the dev extras: pip install -e '.[dev]'", file=sys.stderr)
        return 1
    if not LOCAL_ENDPOINT:
        try:
            import moto  # noqa: F401
        except ImportError:
            print(
                "no DynamoDB backend: either install the dev extras for moto, or set\n"
                "AWS_ENDPOINT_URL_DYNAMODB and run `make local-up` first.",
                file=sys.stderr,
            )
            return 1

    with dynamodb_backend() as (dynamodb, backend_label), demo_tables(dynamodb) as (
        budget_table,
        cache_table,
    ):
        provider = EchoProvider()
        store = BudgetStore(table=budget_table)
        # A deliberately tiny limit so the guardrail trips inside the demo, and a
        # lenient threshold suited to the echo provider's crude embeddings.
        store.put_key(API_KEY, key_id="demo", monthly_limit_usd=0.003, label="demo")
        cache = SemanticCache(
            table=cache_table, embed_fn=provider.embed, threshold=0.80, semantic_enabled=True
        )
        handler_module.set_components(provider=provider, budget=store, cache=cache)

        print("=" * 74)
        print("llm-gateway demo — echo provider, $0.00 real spend")
        print(f"dynamodb: {backend_label}")
        print("=" * 74)

        call(
            "1. cold request: miss, goes upstream, cost accounted",
            event("POST", "/v1/chat", chat(REFUND_Q)),
        )
        call(
            "2. same request: tier-1 exact hit, $0, provider not called",
            event("POST", "/v1/chat", chat(REFUND_Q)),
        )
        call(
            "3. paraphrase: tier-2 semantic hit via LSH + cosine rescoring",
            event("POST", "/v1/chat", chat("How long is the refund window for orders online?")),
        )
        call(
            "4. temperature > 0 bypasses the cache (the caller asked for variety)",
            event("POST", "/v1/chat", chat(REFUND_Q, temperature=0.7)),
        )

        calls_before = provider.calls
        print(f"\n--- 5. spending down the $0.003 monthly limit ({len(FILLER)} distinct questions)")
        for i, question in enumerate(FILLER, start=1):
            status, body = invoke(event("POST", "/v1/chat", chat(question)))
            if status == 429:
                print(f"    blocked at request {i}: {body['error']}")
                print(f"    spend ${body['spend_usd']:.6f} of ${body['limit_usd']:.3f} limit")
                print(f"    {provider.calls - calls_before} requests reached the provider first")
                print(
                    "    note: admission reserves the worst case (max_tokens at the output"
                    "\n    rate), so the gate closes while actual spend is still below the"
                    "\n    limit. That headroom is what makes an overshoot impossible."
                )
                break
        else:
            print("    limit not reached — raise the filler count or lower the limit")

        call(
            "6. cache hits still served with the budget exhausted (they cost nothing)",
            event("POST", "/v1/chat", chat(REFUND_Q)),
        )

        _, usage = invoke(event("GET", "/v1/usage"))
        print("\n--- 7. GET /usage")
        print(f"    period       {usage['period']}")
        print(f"    requests     {usage['requests']}")
        print(f"    cache hits   {usage['cache_hits']}")
        print(f"    spend        ${usage['spend_usd']:.6f}")
        print(f"    tokens       {usage['tokens_in']} in / {usage['tokens_out']} out")

        handler_module.reset_components()

    hits = sum(m["CacheHit"] for m in metrics)
    print(f"\n--- 8. telemetry: {len(metrics)} EMF records emitted, {hits} marked as cache hits")
    print("    one record, exactly as CloudWatch would ingest it:\n")
    sample = next(m for m in metrics if m["CacheHit"] == 1)
    print("    " + json.dumps({k: v for k, v in sample.items() if k != "_aws"}))
    print(f"    + _aws envelope declaring {len(sample['_aws']['CloudWatchMetrics'][0]['Metrics'])} "
          "metrics in namespace LLMGateway")

    print("\n" + "=" * 74)
    print("Nothing was deployed and no provider was called. `make bench` measures")
    print("the cache's hit rate, precision and net saving across thresholds.")
    print("=" * 74 + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
