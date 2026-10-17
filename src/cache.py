"""
Two-tier response cache: exact match, then semantic match.

Tier 1) SHA-256 over (provider, model, system, messages,
temperature, max_tokens). Free, instant, and it catches the retry storms and
duplicated prompts that make up a surprising share of real traffic.

Tier 2) semantic. "What is the refund window?" and "how long do I have to
ask for a refund?" should share one answer. That needs vector similarity, and
DynamoDB has no vector index so the cache brings its own, using
--random-hyperplane LSH--.

    p_match = (1 - θ/π) ** k

Real paraphrases sit around cosine 0.85 (θ ≈ 0.555 rad, so 1 - θ/π ≈ 0.823). At
k=16 that is 0.823**16 ≈ 4.5% — the cache would miss almost every paraphrase it
exists to catch. Widening the search by probing neighbouring buckets scales
badly: covering 3 bit-flips of a 16-bit id means 697 lookups. -- AI reference

The standard fix is L independent tables of k bits each — the OR half of an
AND/OR construction. A candidate is recovered if any table agrees:

    recall = 1 - (1 - p_match) ** L
--------AI Reference------------------
With k=8 and L=12: p_match = 0.823**8 ≈ 0.211, recall = 1 - 0.789**12 ≈ 0.94,
and each bucket still holds only ~1/256 of the cache. Twelve small queries, 94%
recall at cosine 0.85 — and 99.9% at cosine 0.95, which is where a real
embedding model puts most true paraphrases. `k` controls how selective the
shortlist is, `L` controls recall; both are deployment parameters, and
`scripts/lsh_recall.py` prints the table for any pair.

LSH only shortlists. The final decision is an exact cosine comparison against
threshold, so a bad bucket guess can cost a miss but never a wrong answer.

Two more deliberate constraints:

*The hyperplanes must never change. They are derived from a fixed seed, so
  every Lambda instance and every future deploy computes identical buckets. The
  seed and geometry are part of the stored bucket key, so changing them retires
  old entries instead of silently mixing incompatible bucket spaces.
* Only deterministic requests are cached. temperature > 0 means the caller
  asked for variety; serving a stored answer would quietly take it away.

Storage layout keeps the vectors in one place. The exact row holds the answer and
its embedding; each band row is a tiny pointer back to it. One cached response
is therefore 1 fat row + L thin rows, not L copies of a 1,024-float vector.

    pk = "exact#<sha256>"                      sk = "v1"      -> text, tokens, embedding
    pk = "lsh#<seed>.<L>x<k>#<model>#<band>#<sig>"  sk = "<sha256>"  -> pointer only
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
import time
from dataclasses import dataclass
from decimal import Decimal

from .providers.base import ChatRequest

DEFAULT_BANDS = 12
DEFAULT_BITS = 8
DEFAULT_SEED = 1337
DEFAULT_THRESHOLD = 0.92
DEFAULT_TTL_SECONDS = 7 * 24 * 3600
MAX_PER_BUCKET = 8
MAX_CANDIDATES = 32


@dataclass(frozen=True)
class CacheHit:
    text: str
    model: str
    input_tokens: int
    output_tokens: int
    tier: str  # "exact" | "semantic"
    similarity: float


def exact_key(provider: str, request: ChatRequest) -> str:
    payload = json.dumps(
        {
            "p": provider,
            "m": request.model,
            "s": request.system,
            "msgs": request.messages,
            "t": request.temperature,
            "mt": request.max_tokens,
        },
        sort_keys=True,
        ensure_ascii=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def cosine(a: list[float], b: list[float]) -> float:
    if len(a) != len(b):
        return -1.0
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0.0 or nb == 0.0:
        return -1.0
    return dot / (na * nb)


def expected_recall(cosine_similarity: float, bands: int, bits: int) -> float:
    """Probability that LSH shortlists a pair at this similarity.

    Exposed because it is the number that justifies the bands/bits defaults,
    and because it lets anyone re-derive them for their own threshold rather
    than inheriting two magic constants.
    """
    theta = math.acos(max(-1.0, min(1.0, cosine_similarity)))
    p_bit = 1.0 - theta / math.pi
    p_band = p_bit**bits
    return 1.0 - (1.0 - p_band) ** bands


class LSHIndex:
    """Multi-band random-hyperplane LSH over embedding vectors."""

    def __init__(
        self, bands: int = DEFAULT_BANDS, bits: int = DEFAULT_BITS, seed: int = DEFAULT_SEED
    ) -> None:
        self.bands = bands
        self.bits = bits
        self.seed = seed
        self._planes: list[list[list[float]]] | None = None
        self._dim: int | None = None

    @property
    def geometry(self) -> str:
        return f"{self.seed}.{self.bands}x{self.bits}"

    def _ensure_planes(self, dim: int) -> list[list[list[float]]]:
        # Built on first use: the embedding dimension belongs to the provider,
        # not to our config. Seeded, so identical in every container.
        if self._planes is None or self._dim != dim:
            rng = random.Random(self.seed)
            self._planes = [
                [[rng.gauss(0.0, 1.0) for _ in range(dim)] for _ in range(self.bits)]
                for _ in range(self.bands)
            ]
            self._dim = dim
        return self._planes

    def signatures(self, vector: list[float]) -> list[str]:
        """One bucket id per band, in band order."""
        planes = self._ensure_planes(len(vector))
        out = []
        for band in planes:
            bits = [
                "1" if sum(v * p for v, p in zip(vector, plane, strict=True)) >= 0 else "0"
                for plane in band
            ]
            out.append("".join(bits))
        return out

    def expected_recall(self, cosine_similarity: float) -> float:
        return expected_recall(cosine_similarity, self.bands, self.bits)


class SemanticCache:
    """DynamoDB-backed cache. `embed_fn` is injected so the cache never imports a provider."""

    def __init__(
        self,
        table=None,
        table_name: str | None = None,
        embed_fn=None,
        bands: int | None = None,
        bits: int | None = None,
        seed: int | None = None,
        threshold: float | None = None,
        ttl_seconds: int | None = None,
        semantic_enabled: bool | None = None,
    ) -> None:
        self._table = table
        self._table_name = table_name or os.environ.get("CACHE_TABLE", "llm-gateway-cache")
        self.embed_fn = embed_fn
        self.threshold = _env_float("CACHE_THRESHOLD", DEFAULT_THRESHOLD, threshold)
        self.ttl_seconds = int(_env_float("CACHE_TTL_SECONDS", DEFAULT_TTL_SECONDS, ttl_seconds))
        self.semantic_enabled = (
            semantic_enabled
            if semantic_enabled is not None
            else os.environ.get("CACHE_SEMANTIC", "1") == "1"
        )
        self.index = LSHIndex(
            bands=int(_env_float("CACHE_LSH_BANDS", DEFAULT_BANDS, bands)),
            bits=int(_env_float("CACHE_LSH_BITS", DEFAULT_BITS, bits)),
            seed=int(_env_float("CACHE_LSH_SEED", DEFAULT_SEED, seed)),
        )

    @property
    def table(self):
        if self._table is None:
            import boto3

            self._table = boto3.resource(
                "dynamodb", region_name=os.environ.get("AWS_REGION", "us-east-1")
            ).Table(self._table_name)
        return self._table

    def _band_pk(self, model: str, band: int, signature: str) -> str:
        return f"lsh#{self.index.geometry}#{model}#{band}#{signature}"

    def cacheable(self, request: ChatRequest) -> bool:
        return request.temperature == 0.0

    def _load_exact(self, key: str) -> dict | None:
        return self.table.get_item(Key={"pk": f"exact#{key}", "sk": "v1"}).get("Item")

    @staticmethod
    def _as_hit(item: dict, fallback_model: str, tier: str, similarity: float) -> CacheHit:
        return CacheHit(
            text=item["text"],
            model=item.get("model", fallback_model),
            input_tokens=int(item.get("input_tokens", 0)),
            output_tokens=int(item.get("output_tokens", 0)),
            tier=tier,
            similarity=similarity,
        )

    # ------------------------------------------------------------------ read

    def get(self, provider: str, request: ChatRequest) -> CacheHit | None:
        if not self.cacheable(request):
            return None

        item = self._load_exact(exact_key(provider, request))
        if item:
            return self._as_hit(item, request.model, "exact", 1.0)

        if not (self.semantic_enabled and self.embed_fn):
            return None

        vector = self.embed_fn(request.prompt_text())

        # Shortlist: union the band buckets, deduplicated and capped so a hot
        # bucket cannot turn one lookup into hundreds of reads.
        candidate_keys: list[str] = []
        seen: set[str] = set()
        for band, signature in enumerate(self.index.signatures(vector)):
            resp = self.table.query(
                KeyConditionExpression="pk = :pk",
                ExpressionAttributeValues={":pk": self._band_pk(request.model, band, signature)},
                Limit=MAX_PER_BUCKET,
            )
            for row in resp.get("Items", []):
                key = row["sk"]
                if key not in seen:
                    seen.add(key)
                    candidate_keys.append(key)
            if len(candidate_keys) >= MAX_CANDIDATES:
                break

        # Rescore: LSH proposed, cosine decides.
        best: tuple[float, dict] | None = None
        for key in candidate_keys[:MAX_CANDIDATES]:
            candidate = self._load_exact(key)
            if not candidate:
                continue  # the pointer outlived its row; TTL will tidy it up
            score = cosine(vector, [float(x) for x in candidate.get("embedding", [])])
            if score >= self.threshold and (best is None or score > best[0]):
                best = (score, candidate)

        if best is None:
            return None
        score, candidate = best
        return self._as_hit(candidate, request.model, "semantic", round(score, 4))

    # ----------------------------------------------------------------- write

    def put(
        self,
        provider: str,
        request: ChatRequest,
        text: str,
        input_tokens: int,
        output_tokens: int,
    ) -> None:
        if not self.cacheable(request):
            return

        expires_at = int(time.time()) + self.ttl_seconds
        key = exact_key(provider, request)
        item = {
            "pk": f"exact#{key}",
            "sk": "v1",
            "text": text,
            "model": request.model,
            "input_tokens": int(input_tokens),
            "output_tokens": int(output_tokens),
            "expires_at": expires_at,
        }

        if self.semantic_enabled and self.embed_fn:
            vector = self.embed_fn(request.prompt_text())
            item["embedding"] = [Decimal(str(round(x, 6))) for x in vector]
            self.table.put_item(Item=item)
            # Band rows carry no vector — one fat row, L thin pointers.
            for band, signature in enumerate(self.index.signatures(vector)):
                self.table.put_item(
                    Item={
                        "pk": self._band_pk(request.model, band, signature),
                        "sk": key,
                        "expires_at": expires_at,
                    }
                )
        else:
            self.table.put_item(Item=item)


def _env_float(name: str, default: float, override: float | None) -> float:
    if override is not None:
        return float(override)
    return float(os.environ.get(name, default))
