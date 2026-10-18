#!/usr/bin/env python3
"""Prepare DynamoDB Local: wait for it, create the tables, optionally seed a key.

Idempotent — run it as often as you like. Safe by construction: it refuses to
run against real AWS, because a script whose job is "create tables and seed a
key" pointed at a production account is exactly the kind of accident that is
obvious only afterwards.

    make local-up && make local-init

    python scripts/local_setup.py --seed-key dev --limit 5
"""

from __future__ import annotations

import argparse
import os
import pathlib
import secrets
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from src.budget import BudgetStore  # noqa: E402
from src.tables import create_table  # noqa: E402

DEFAULT_ENDPOINT = "http://localhost:8000"
LOCAL_HOSTS = ("localhost", "127.0.0.1", "0.0.0.0", "dynamodb", "host.docker.internal")


def guard_is_local(endpoint: str) -> None:
    """Refuse to touch anything that is not obviously a local endpoint."""
    from urllib.parse import urlparse

    host = (urlparse(endpoint).hostname or "").lower()
    if host not in LOCAL_HOSTS:
        raise SystemExit(
            f"refusing to run against {endpoint!r}: host {host!r} is not local.\n"
            "This script creates tables and seeds credentials; point it at "
            "DynamoDB Local only."
        )


def wait_for_dynamodb(client, attempts: int = 30, delay: float = 1.0) -> None:
    """Poll the API until it answers.

    Waiting on the API rather than on the TCP port matters: the container
    accepts connections a moment before DynamoDB is ready to serve, and the
    gap produces a confusing first-run failure.
    """
    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            client.list_tables()
            if attempt > 1:
                print(f"DynamoDB Local answered after {attempt} attempts")
            return
        except Exception as exc:  # noqa: BLE001 - any failure means not ready yet
            last_error = exc
            if attempt == 1:
                print("waiting for DynamoDB Local", end="", flush=True)
            else:
                print(".", end="", flush=True)
            time.sleep(delay)
    print()
    raise SystemExit(
        f"DynamoDB Local did not answer after {attempts} attempts: {last_error}\n"
        "Is it running? Try: make local-up"
    )


def ensure_table(resource, client, name: str) -> str:
    existing = client.list_tables().get("TableNames", [])
    if name in existing:
        return "exists"
    create_table(resource, name)
    # DynamoDB Local creates tables synchronously, so no waiter is needed here.
    # Against real AWS you would wait on the table_exists waiter.
    return "created"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--endpoint",
        default=os.environ.get("AWS_ENDPOINT_URL_DYNAMODB", DEFAULT_ENDPOINT),
    )
    parser.add_argument(
        "--budget-table", default=os.environ.get("BUDGET_TABLE", "llm-gateway-budgets")
    )
    parser.add_argument(
        "--cache-table", default=os.environ.get("CACHE_TABLE", "llm-gateway-cache")
    )
    parser.add_argument("--seed-key", default=None, help="Key id to mint, e.g. dev")
    parser.add_argument("--limit", type=float, default=5.0, help="Monthly cap for the seeded key")
    parser.add_argument("--reset", action="store_true", help="Delete the tables first")
    parser.add_argument(
        "--write-env",
        default=".env.local",
        help="File to write the shell exports and the minted key into (gitignored)",
    )
    args = parser.parse_args()

    guard_is_local(args.endpoint)

    import boto3

    # Explicit dummy credentials, never the ambient ones. Two reasons, both
    # learned the hard way:
    #
    # 1. Whatever is in your shell might be real. This script creates tables and
    #    seeds an API key; it must not be able to do that to an AWS account
    #    because a profile happened to be exported.
    # 2. DynamoDB Local validates the *shape* of an access key and rejects
    #    values that do not look like one. Inheriting a placeholder from some
    #    other tool produces `UnrecognizedClientException`, which reads like a
    #    permissions problem and is not one.
    #
    # `local` is accepted, deterministic, and obviously not a real credential.
    session = boto3.Session(
        aws_access_key_id="local",
        aws_secret_access_key="local",
        aws_session_token=None,
        region_name="us-east-1",
    )
    client = session.client("dynamodb", endpoint_url=args.endpoint)
    resource = session.resource("dynamodb", endpoint_url=args.endpoint)

    print(f"endpoint   {args.endpoint}")
    wait_for_dynamodb(client)

    if args.reset:
        for name in (args.budget_table, args.cache_table):
            if name in client.list_tables().get("TableNames", []):
                client.delete_table(TableName=name)
                print(f"deleted    {name}")

    for name in (args.budget_table, args.cache_table):
        print(f"{ensure_table(resource, client, name):<10} {name}")

    api_key = None
    if args.seed_key:
        api_key = f"gw_local_{secrets.token_urlsafe(16)}"
        store = BudgetStore(table=resource.Table(args.budget_table))
        store.put_key(
            api_key, key_id=args.seed_key, monthly_limit_usd=args.limit, label=args.seed_key
        )
        print(f"\nkey id     {args.seed_key}")
        print(f"limit      ${args.limit:.2f}/month")
        print(f"api key    {api_key}")
        print("           ^ stored hashed; recoverable only from the file below")

    lines = [
        "# Written by scripts/local_setup.py. Local development only.",
        "# Gitignored: it holds a (local, worthless) API key.",
        f"export AWS_ENDPOINT_URL_DYNAMODB={args.endpoint}",
        f"export BUDGET_TABLE={args.budget_table}",
        f"export CACHE_TABLE={args.cache_table}",
        "export AWS_ACCESS_KEY_ID=local",
        "export AWS_SECRET_ACCESS_KEY=local",
        "export AWS_DEFAULT_REGION=us-east-1",
        "export LLM_PROVIDER=echo  # no model calls, no spend",
    ]
    if api_key:
        lines.append(f"export GATEWAY_API_KEY={api_key}")

    if args.write_env:
        path = pathlib.Path(args.write_env)
        if path.exists() and not api_key:
            # Don't clobber a file holding a key just because this run didn't
            # mint one — that would silently break `make local-smoke`.
            print(f"\nkept       {path} (no new key minted this run)")
        else:
            path.write_text("\n".join(lines) + "\n", encoding="utf-8")
            print(f"\nwrote      {path}")

    print("\nTo use it in your shell:\n")
    print(f"  source {args.write_env}\n" if args.write_env else "")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
