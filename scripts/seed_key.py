#!/usr/bin/env python3
"""Mint an API key with a monthly budget against a deployed stack.

The key is generated here, printed once, and stored only as a SHA-256 hash — so
if you lose it, mint another; it cannot be recovered from the table.

"""

from __future__ import annotations

import argparse
import secrets
import sys

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))

from src.budget import BudgetStore  # noqa: E402


def stack_output(stack: str, region: str, key: str) -> str:
    import boto3

    cfn = boto3.client("cloudformation", region_name=region)
    outputs = cfn.describe_stacks(StackName=stack)["Stacks"][0]["Outputs"]
    for output in outputs:
        if output["OutputKey"] == key:
            return output["OutputValue"]
    raise SystemExit(f"stack {stack} has no output named {key}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stack", default="llm-gateway")
    parser.add_argument("--region", default="us-east-1")
    parser.add_argument("--limit", type=float, required=True, help="Monthly cap in USD")
    parser.add_argument("--label", default="dev")
    parser.add_argument("--key-id", default=None, help="Defaults to the label")
    parser.add_argument("--table", default=None, help="Skip stack lookup and use this table")
    args = parser.parse_args()

    import boto3

    table_name = args.table or stack_output(args.stack, args.region, "BudgetTableName")
    table = boto3.resource("dynamodb", region_name=args.region).Table(table_name)

    api_key = f"gw_{secrets.token_urlsafe(24)}"
    store = BudgetStore(table=table)
    store.put_key(
        api_key,
        key_id=args.key_id or args.label,
        monthly_limit_usd=args.limit,
        label=args.label,
    )

    api_url = stack_output(args.stack, args.region, "ApiUrl") if not args.table else "<api-url>"
    print(f"\nkey id     {args.key_id or args.label}")
    print(f"limit      ${args.limit:.2f}/month")
    print(f"api key    {api_key}")
    print("           ^ stored hashed; this is the only time it is shown\n")
    print("Try it:\n")
    print(f'  curl -s {api_url}/chat \\')
    print(f'    -H "Authorization: Bearer {api_key}" \\')
    print('    -H "Content-Type: application/json" \\')
    print(
        '    -d \'{"model":"anthropic.claude-3-5-haiku-20241022-v1:0",'
        '"messages":[{"role":"user","content":"say hi"}]}\' | jq\n'
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
