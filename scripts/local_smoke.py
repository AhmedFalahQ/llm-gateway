#!/usr/bin/env python3
"""Drive the gateway against the persistent local tables, as a real client would.

`bench/demo.py` creates throwaway tables and its own key, so nothing it does
survives the run. This does the opposite: it uses the tables and the API key
that `local_setup.py` created, so afterwards `make local-usage` has something
to report and the ledger accumulates across runs — which is what you want when
checking that the monthly spend record behaves over time rather than inside one
process.

    make local-init && make local-smoke && make local-usage

Reads configuration from the environment, or from `.env.local` if present.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

ENV_FILE = pathlib.Path(".env.local")


def load_env_file(path: pathlib.Path) -> None:
    """Apply `export K=V` lines without overriding what is already set."""
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        line = line.removeprefix("export ").strip()
        key, _, value = line.partition("=")
        if not key or not _:
            continue
        # Strip a trailing inline comment, then surrounding quotes.
        value = value.split("  #")[0].strip().strip("'\"")
        os.environ.setdefault(key.strip(), value)


QUESTIONS = [
    # The first three are the same question in different words. On the echo
    # provider's crude embeddings only some of them will hit, which is the point:
    # the hit rate here is a property of the embedding model, not a promise.
    "What is the refund window for online orders?",
    "What is the refund window for online orders?",
    "How long is the refund window for orders placed online?",
    "How many days does standard shipping take?",
    "What is the API rate limit per minute?",
]


def event(api_key: str, body: dict | None, method: str = "POST", path: str = "/v1/chat") -> dict:
    return {
        "httpMethod": method,
        "path": path,
        "headers": {"Authorization": f"Bearer {api_key}"},
        "body": json.dumps(body) if body is not None else None,
    }


def invoke(handler_module, ev: dict, metrics: list[dict]) -> tuple[int, dict]:
    """Call the handler, keeping its EMF metric lines out of the transcript.

    In Lambda those lines are the point — CloudWatch parses them into metrics.
    On a terminal they bury the output, so they are collected and summarised at
    the end instead.
    """
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        result = handler_module.handler(ev)
    for line in buffer.getvalue().splitlines():
        if line.strip():
            with contextlib.suppress(json.JSONDecodeError):
                metrics.append(json.loads(line))
    return result["statusCode"], json.loads(result["body"])


def main() -> int:
    load_env_file(ENV_FILE)

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api-key", default=os.environ.get("GATEWAY_API_KEY"))
    parser.add_argument("--model", default="anthropic.claude-3-5-haiku-20241022-v1:0")
    parser.add_argument("--repeat", type=int, default=1, help="Passes over the question set")
    args = parser.parse_args()

    if not os.environ.get("AWS_ENDPOINT_URL_DYNAMODB"):
        print(
            "AWS_ENDPOINT_URL_DYNAMODB is not set — this script talks to DynamoDB Local.\n"
            "Run `make local-up && make local-init` first, or use `make local-smoke`.",
            file=sys.stderr,
        )
        return 2
    if not args.api_key:
        print(
            "No API key. Run `make local-init` to mint one into .env.local, "
            "or pass --api-key.",
            file=sys.stderr,
        )
        return 2

    # Imported after the environment is settled: the gateway reads table names
    # and the endpoint at client-construction time.
    from src import handler as handler_module

    print(f"endpoint   {os.environ['AWS_ENDPOINT_URL_DYNAMODB']}")
    print(f"budgets    {os.environ.get('BUDGET_TABLE')}")
    print(f"provider   {os.environ.get('LLM_PROVIDER', 'bedrock')}\n")

    header = f"{'#':>3}  {'status':>6}  {'tier':<9}  {'cost':>10}  question"
    print(header)
    print("-" * (len(header) + 20))

    metrics: list[dict] = []
    count = 0
    for _ in range(args.repeat):
        for question in QUESTIONS:
            count += 1
            status, body = invoke(
                handler_module,
                event(
                    args.api_key,
                    {"model": args.model, "messages": [{"role": "user", "content": question}]},
                ),
                metrics,
            )
            tier = body.get("cache_tier", "-")
            cost = body.get("cost_usd")
            cost_text = f"${cost:.6f}" if isinstance(cost, (int, float)) else "-"
            if status != 200:
                tier, cost_text = "error", body.get("error", "")[:28]
            print(f"{count:>3}  {status:>6}  {tier:<9}  {cost_text:>10}  {question[:44]}")

    _, usage = invoke(
        handler_module, event(args.api_key, None, method="GET", path="/v1/usage"), metrics
    )
    print(
        f"\n{usage['requests']} requests, {usage['cache_hits']} cache hits, "
        f"${usage['spend_usd']:.6f} spent in {usage['period']}"
    )
    print(f"{len(metrics)} EMF metric records emitted (CloudWatch would ingest these)")
    print("\nThis spend persists. `make local-usage` reports it; `make local-reset` clears it.\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
