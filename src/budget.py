"""Per-key cost guardrails backed by DynamoDB.

The hard part of a budget is not the arithmetic, it is concurrency: two Lambda
invocations for the same API key can check the budget at the same instant and
both decide there is room. So the check and the increment have to be a single
operation.

This module does that with one conditional `UpdateItem`:

    ADD spend_usd :estimate
    IF  spend_usd <= (limit - :estimate)   -- or the attribute does not exist yet

DynamoDB evaluates the condition and applies the increment atomically, so the
budget can be overshot by at most one in-flight request's estimate. Because the
estimate is charged up front, every request is then *reconciled* once the
provider reports real usage — a second `ADD` of the difference, which may be
negative. Reconciling in a `finally` block matters: a request that fails after
the reserve must give the money back.

Table layout (single table, two record shapes):

    pk = "apikey#<sha256(api_key)>"  sk = "meta"            -> key_id, limit, enabled
    pk = "spend#<key_id>"            sk = "<YYYY-MM>"       -> spend_usd, tokens, counters
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal

CENTS = Decimal("0.000001")


def hash_api_key(api_key: str) -> str:
    """Keys are stored hashed so the table never holds a usable credential."""
    return hashlib.sha256(api_key.encode("utf-8")).hexdigest()


def current_period(now: datetime | None = None) -> str:
    now = now or datetime.now(UTC)
    return now.strftime("%Y-%m")


class BudgetExceeded(Exception):
    """Raised when a request would push a key past its monthly limit."""

    def __init__(self, key_id: str, limit_usd: float, spend_usd: float) -> None:
        super().__init__(
            f"key {key_id} is at ${spend_usd:.4f} of its ${limit_usd:.2f} monthly limit"
        )
        self.key_id = key_id
        self.limit_usd = limit_usd
        self.spend_usd = spend_usd


class UnknownApiKey(Exception):
    """Raised when the presented key is absent or disabled."""


@dataclass(frozen=True)
class KeyRecord:
    key_id: str
    label: str
    monthly_limit_usd: float
    enabled: bool


@dataclass(frozen=True)
class Reservation:
    """What a successful reserve returns; hand it back to `reconcile`."""

    key_id: str
    period: str
    estimated_usd: float
    spend_after_usd: float

    def remaining_usd(self, limit_usd: float) -> float:
        return max(0.0, limit_usd - self.spend_after_usd)


def _to_decimal(value: float) -> Decimal:
    return Decimal(str(value)).quantize(CENTS)


class BudgetStore:
    """DynamoDB-backed key directory and spend ledger."""

    def __init__(self, table=None, table_name: str | None = None) -> None:
        self._table = table
        self._table_name = table_name or os.environ.get("BUDGET_TABLE", "llm-gateway-budgets")

    @property
    def table(self):
        if self._table is None:
            import boto3

            self._table = boto3.resource(
                "dynamodb", region_name=os.environ.get("AWS_REGION", "us-east-1")
            ).Table(self._table_name)
        return self._table

    # ------------------------------------------------------------------ keys

    def put_key(
        self, api_key: str, key_id: str, monthly_limit_usd: float, label: str = ""
    ) -> KeyRecord:
        """Register an API key. Used by `make seed-key`, not by the hot path."""
        self.table.put_item(
            Item={
                "pk": f"apikey#{hash_api_key(api_key)}",
                "sk": "meta",
                "key_id": key_id,
                "label": label or key_id,
                "monthly_limit_usd": _to_decimal(monthly_limit_usd),
                "enabled": True,
            }
        )
        return KeyRecord(key_id, label or key_id, monthly_limit_usd, True)

    def lookup_key(self, api_key: str) -> KeyRecord:
        resp = self.table.get_item(Key={"pk": f"apikey#{hash_api_key(api_key)}", "sk": "meta"})
        item = resp.get("Item")
        if not item:
            raise UnknownApiKey("api key not recognized")
        if not item.get("enabled", False):
            raise UnknownApiKey("api key is disabled")
        return KeyRecord(
            key_id=item["key_id"],
            label=item.get("label", item["key_id"]),
            monthly_limit_usd=float(item["monthly_limit_usd"]),
            enabled=True,
        )

    # ---------------------------------------------------------------- spend

    def reserve(
        self, key: KeyRecord, estimated_usd: float, period: str | None = None
    ) -> Reservation:
        """Atomically admit a request and charge `estimated_usd` against the key.

        Raises `BudgetExceeded` if there is not enough headroom.
        """
        period = period or current_period()
        estimate = _to_decimal(estimated_usd)
        headroom = _to_decimal(key.monthly_limit_usd) - estimate

        try:
            resp = self.table.update_item(
                Key={"pk": f"spend#{key.key_id}", "sk": period},
                UpdateExpression=(
                    "ADD spend_usd :est, requests :one "
                    "SET key_id = if_not_exists(key_id, :kid)"
                ),
                ConditionExpression=(
                    "attribute_not_exists(spend_usd) OR spend_usd <= :headroom"
                ),
                ExpressionAttributeValues={
                    ":est": estimate,
                    ":one": 1,
                    ":headroom": headroom,
                    ":kid": key.key_id,
                },
                ReturnValues="UPDATED_NEW",
            )
        except Exception as exc:  # noqa: BLE001 - narrowed below
            if type(exc).__name__ == "ConditionalCheckFailedException" or (
                "ConditionalCheckFailed" in str(exc)
            ):
                raise BudgetExceeded(
                    key.key_id, key.monthly_limit_usd, self.spend(key.key_id, period)
                ) from None
            raise

        attributes = resp.get("Attributes", {})
        spend_after = (
            float(attributes["spend_usd"])
            if "spend_usd" in attributes
            else self.spend(key.key_id, period)
        )
        return Reservation(key.key_id, period, float(estimate), spend_after)

    def reconcile(
        self,
        reservation: Reservation,
        actual_usd: float,
        input_tokens: int = 0,
        output_tokens: int = 0,
        cache_hit: bool = False,
    ) -> float:
        """Replace the reserved estimate with the real cost.

        Call this exactly once per reservation, including on failure (with
        `actual_usd=0.0`), or keys slowly leak budget to requests that never ran.
        """
        delta = _to_decimal(actual_usd) - _to_decimal(reservation.estimated_usd)
        resp = self.table.update_item(
            Key={"pk": f"spend#{reservation.key_id}", "sk": reservation.period},
            UpdateExpression=(
                "ADD spend_usd :delta, tokens_in :ti, tokens_out :to_, cache_hits :ch"
            ),
            ExpressionAttributeValues={
                ":delta": delta,
                ":ti": int(input_tokens),
                ":to_": int(output_tokens),
                ":ch": 1 if cache_hit else 0,
            },
            ReturnValues="UPDATED_NEW",
        )
        # An `ADD` of zero can be elided from UPDATED_NEW — which happens
        # whenever the estimate was exactly right — so fall back to a read
        # rather than trusting the attribute to be there.
        attributes = resp.get("Attributes", {})
        if "spend_usd" in attributes:
            return float(attributes["spend_usd"])
        return self.spend(reservation.key_id, reservation.period)

    def record_hit(self, key_id: str, period: str | None = None) -> None:
        """Count a cache hit.

        Hits are served without a reservation — they cost nothing upstream, so a
        key that has exhausted its budget can still be served from cache — but
        they still belong in the counters, or the hit rate looks like zero.
        """
        period = period or current_period()
        self.table.update_item(
            Key={"pk": f"spend#{key_id}", "sk": period},
            UpdateExpression=(
                "ADD requests :one, cache_hits :one SET key_id = if_not_exists(key_id, :kid)"
            ),
            ExpressionAttributeValues={":one": 1, ":kid": key_id},
        )

    def spend(self, key_id: str, period: str | None = None) -> float:
        period = period or current_period()
        resp = self.table.get_item(Key={"pk": f"spend#{key_id}", "sk": period})
        item = resp.get("Item") or {}
        return float(item.get("spend_usd", 0))

    def usage(self, key_id: str, period: str | None = None) -> dict:
        """Everything the `/v1/usage` endpoint reports for one key."""
        period = period or current_period()
        item = self.table.get_item(Key={"pk": f"spend#{key_id}", "sk": period}).get("Item") or {}
        return {
            "key_id": key_id,
            "period": period,
            "spend_usd": round(float(item.get("spend_usd", 0)), 6),
            "requests": int(item.get("requests", 0)),
            "cache_hits": int(item.get("cache_hits", 0)),
            "tokens_in": int(item.get("tokens_in", 0)),
            "tokens_out": int(item.get("tokens_out", 0)),
        }
