#!/usr/bin/env python3
"""Print this period's spend for every key in a deployed stack.

"""

from __future__ import annotations

import argparse
import sys

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))

from src.budget import current_period  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stack", default="llm-gateway")
    parser.add_argument("--region", default="us-east-1")
    parser.add_argument("--period", default=None, help="YYYY-MM, defaults to the current month")
    parser.add_argument("--table", default=None)
    args = parser.parse_args()

    import boto3

    table_name = args.table
    if not table_name:
        cfn = boto3.client("cloudformation", region_name=args.region)
        outputs = cfn.describe_stacks(StackName=args.stack)["Stacks"][0]["Outputs"]
        table_name = next(
            o["OutputValue"] for o in outputs if o["OutputKey"] == "BudgetTableName"
        )

    table = boto3.resource("dynamodb", region_name=args.region).Table(table_name)
    period = args.period or current_period()

    # A scan is the right tool here: the table holds one row per key per month,
    # so it is tiny, and this runs from a laptop rather than the hot path.
    rows = [
        item
        for item in table.scan().get("Items", [])
        if str(item.get("pk", "")).startswith("spend#") and item.get("sk") == period
    ]
    rows.sort(key=lambda r: float(r.get("spend_usd", 0)), reverse=True)

    header = (
        f"{'key':<20} {'spend':>10} {'reqs':>7} {'hits':>7} "
        f"{'hit %':>7} {'tok in':>9} {'tok out':>9}"
    )
    print(f"\nperiod {period}\n")
    print(header)
    print("-" * len(header))
    total = 0.0
    for row in rows:
        spend = float(row.get("spend_usd", 0))
        total += spend
        reqs = int(row.get("requests", 0))
        hits = int(row.get("cache_hits", 0))
        rate = f"{hits / reqs * 100:.0f}%" if reqs else "-"
        print(
            f"{str(row.get('key_id', row['pk'])):<20} ${spend:>9.4f} {reqs:>7} {hits:>7} "
            f"{rate:>7} {int(row.get('tokens_in', 0)):>9} {int(row.get('tokens_out', 0)):>9}"
        )
    print("-" * len(header))
    print(f"{'total':<20} ${total:>9.4f}\n")
    if not rows:
        print("(no activity recorded for this period)\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
