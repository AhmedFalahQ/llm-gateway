#!/usr/bin/env python3
"""Cache benchmark and threshold calibrator.

Answers the only two questions that matter about a semantic cache:

1. How much does it save? Replay a query stream and compare what the
   provider would have been paid with the cache off versus on — net of the
   embedding calls the cache itself makes, which is the number most write-ups
   quietly omit.
2. How often is it wrong? Every query in the input is labelled with a
   `group`. Queries in the same group are paraphrases and *should* share an
   answer; a hit across groups is a wrong answer served to a user. That makes
   cache precision measurable rather than a matter of faith.

The two trade off against each other through `--threshold`, so the default run
sweeps a range and prints the curve. Pick the highest hit rate that still holds
precision at 1.00 on your own traffic — and re-run this whenever you change
embedding model, because the right threshold is a property of that model, not of
this code.


"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from src.cache import SemanticCache  # noqa: E402
from src.pricing import cost_usd, estimate_input_tokens, price_for  # noqa: E402
from src.providers.base import ChatRequest  # noqa: E402
from src.providers.echo import EchoProvider  # noqa: E402
from src.tables import create_table  # noqa: E402

# Titan Text Embeddings v2, USD per 1k tokens. Cheap, but not free — and the
# cache pays it on every lookup, so it belongs in the savings arithmetic.
EMBED_PRICE_PER_1K = 0.00002

DEFAULT_THRESHOLDS = [0.80, 0.85, 0.90, 0.92, 0.95, 0.98]


def load_queries(path: pathlib.Path) -> list[dict]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def run_once(queries: list[dict], model: str, threshold: float, dynamodb) -> dict:
    """Replay the stream once at one threshold and return the measurements."""
    table_name = f"bench-cache-{str(threshold).replace('.', '')}"
    table = create_table(dynamodb, table_name)
    provider = EchoProvider()
    cache = SemanticCache(
        table=table, embed_fn=provider.embed, threshold=threshold, semantic_enabled=True
    )

    # Maps a stored answer back to the group it was generated for, which is what
    # makes a cross-group hit detectable.
    answer_group: dict[str, str] = {}

    stats = {
        "threshold": threshold,
        "queries": len(queries),
        "exact_hits": 0,
        "semantic_hits": 0,
        "misses": 0,
        "correct_hits": 0,
        "wrong_hits": 0,
        "provider_cost_no_cache": 0.0,
        "provider_cost_with_cache": 0.0,
        "embed_cost": 0.0,
    }

    for row in queries:
        request = ChatRequest(model=model, messages=[{"role": "user", "content": row["prompt"]}])
        baseline = provider.chat(request)
        stats["provider_cost_no_cache"] += cost_usd(
            model, baseline.input_tokens, baseline.output_tokens
        )

        embed_tokens = estimate_input_tokens(request.prompt_text())
        hit = cache.get(provider.name, request)
        # A lookup embeds once on an exact miss; an exact hit short-circuits
        # before embedding.
        if hit is None or hit.tier == "semantic":
            stats["embed_cost"] += embed_tokens / 1000.0 * EMBED_PRICE_PER_1K

        if hit is not None:
            stats[f"{hit.tier}_hits"] += 1
            if answer_group.get(hit.text) == row["group"]:
                stats["correct_hits"] += 1
            else:
                stats["wrong_hits"] += 1
            continue

        stats["misses"] += 1
        result = provider.chat(request)
        stats["provider_cost_with_cache"] += cost_usd(
            model, result.input_tokens, result.output_tokens
        )
        cache.put(provider.name, request, result.text, result.input_tokens, result.output_tokens)
        # The write embeds again (the read path does not hand its vector over).
        stats["embed_cost"] += embed_tokens / 1000.0 * EMBED_PRICE_PER_1K
        answer_group[result.text] = row["group"]

    hits = stats["exact_hits"] + stats["semantic_hits"]
    stats["hit_rate"] = hits / len(queries) if queries else 0.0
    stats["precision"] = (stats["correct_hits"] / hits) if hits else 1.0
    stats["total_with_cache"] = stats["provider_cost_with_cache"] + stats["embed_cost"]
    stats["net_saved"] = stats["provider_cost_no_cache"] - stats["total_with_cache"]
    stats["saved_pct"] = (
        stats["net_saved"] / stats["provider_cost_no_cache"] * 100.0
        if stats["provider_cost_no_cache"]
        else 0.0
    )
    return stats


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queries", type=pathlib.Path, default="bench/queries.sample.jsonl")
    parser.add_argument("--model", default="anthropic.claude-3-5-haiku-20241022-v1:0")
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Measure one threshold instead of sweeping the default range.",
    )
    parser.add_argument("--json", action="store_true", help="Emit raw JSON instead of a table.")
    args = parser.parse_args()

    queries = load_queries(pathlib.Path(args.queries))
    if not queries:
        print("no queries loaded", file=sys.stderr)
        return 1

    thresholds = [args.threshold] if args.threshold is not None else DEFAULT_THRESHOLDS

    try:
        import boto3
        from moto import mock_aws
    except ImportError:
        print("this benchmark needs the dev extras: pip install -e '.[dev]'", file=sys.stderr)
        return 1

    results = []
    with mock_aws():
        dynamodb = boto3.resource("dynamodb", region_name="us-east-1")
        for threshold in thresholds:
            results.append(run_once(queries, args.model, threshold, dynamodb))

    if args.json:
        print(json.dumps(results, indent=2))
        return 0

    price = price_for(args.model)
    groups = len({q["group"] for q in queries})
    print(
        f"\nmodel      {args.model}"
        f"  (${price.input_per_1k}/1k in, ${price.output_per_1k}/1k out)"
    )
    print(f"queries    {len(queries)} across {groups} paraphrase groups")
    print(f"embeddings ${EMBED_PRICE_PER_1K}/1k tokens, charged on every lookup and write\n")
    header = (
        f"{'thresh':>7}  {'hit rate':>9}  {'precision':>10}  {'exact':>6}  "
        f"{'sem':>5}  {'net saved':>10}  {'saved %':>8}"
    )
    print(header)
    print("-" * len(header))
    for r in results:
        print(
            f"{r['threshold']:>7.2f}  {r['hit_rate']:>8.1%}  {r['precision']:>10.2f}  "
            f"{r['exact_hits']:>6}  {r['semantic_hits']:>5}  ${r['net_saved']:>9.6f}  "
            f"{r['saved_pct']:>7.1f}%"
        )

    clean = [r for r in results if r["precision"] >= 1.0]
    if clean:
        best = max(clean, key=lambda r: r["hit_rate"])
        print(
            f"\nbest threshold with no wrong answers: {best['threshold']:.2f} "
            f"-> {best['hit_rate']:.1%} hit rate, {best['saved_pct']:.1f}% net saving"
        )
    else:
        print("\nno threshold in the sweep achieved precision 1.00 on this query set")
    print(
        "\nNote: measured with the `echo` provider's bag-of-words embeddings. A real\n"
        "embedding model separates paraphrases far better, so re-calibrate against\n"
        "your own traffic before trusting a threshold in production.\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
