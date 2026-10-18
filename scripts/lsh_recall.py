#!/usr/bin/env python3
"""Print the LSH recall table for a given (bands, bits) geometry.

The cache's two tuning knobs are easy to set badly and hard to debug once set,
because a bad geometry looks exactly like "semantic caching just doesn't work".
This prints the probability that a pair at a given cosine similarity makes it
onto the shortlist, so the defaults in `src/cache.py` can be checked rather than
trusted — and re-derived for a different embedding model.
"""

from __future__ import annotations

import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from src.cache import DEFAULT_BANDS, DEFAULT_BITS, expected_recall  # noqa: E402

SIMILARITIES = [0.70, 0.80, 0.85, 0.90, 0.92, 0.95, 0.98, 0.99]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bands", type=int, default=DEFAULT_BANDS)
    parser.add_argument("--bits", type=int, default=DEFAULT_BITS)
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Also show a few alternative geometries side by side.",
    )
    args = parser.parse_args()

    geometries = [(args.bands, args.bits)]
    if args.compare:
        geometries += [(1, 16), (1, 8), (6, 8), (24, 8), (12, 12)]

    print(f"\n{'cosine':>8}", end="")
    for bands, bits in geometries:
        print(f"{f'{bands}x{bits}':>10}", end="")
    print(f"\n{'-' * (8 + 10 * len(geometries))}")

    for similarity in SIMILARITIES:
        print(f"{similarity:>8.2f}", end="")
        for bands, bits in geometries:
            print(f"{expected_recall(similarity, bands, bits):>9.1%} ", end="")
        print()

    buckets = 2 ** args.bits
    print(
        f"\n{args.bands}x{args.bits}: {args.bands} queries per lookup, "
        f"~1/{buckets} of the cache per bucket.\n"
        "Recall is the chance a true match is shortlisted; cosine then decides.\n"
        "Raise bands for recall, raise bits for a tighter shortlist.\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
