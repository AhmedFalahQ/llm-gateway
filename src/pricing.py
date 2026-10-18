"""Token pricing.

Prices are USD per 1,000 tokens and live in one place on purpose: the gateway's
whole value proposition is that it can tell you what a request cost *before* the
monthly bill arrives, and that is only true if the table is easy to keep current.

Every price below is editable config, not a promise. Providers change prices;
re-check them against the provider's own pricing page and bump `PRICE_SHEET_DATE`
when you do. Unknown models fall back to `DEFAULT_PRICE` so the gateway never
silently accounts a request as free.
"""

from __future__ import annotations

from dataclasses import dataclass

PRICE_SHEET_DATE = "2026-10-01"


@dataclass(frozen=True)
class ModelPrice:
    """USD per 1,000 tokens, split by direction."""

    input_per_1k: float
    output_per_1k: float

    def cost_usd(self, input_tokens: int, output_tokens: int) -> float:
        return (
            input_tokens / 1000.0 * self.input_per_1k
            + output_tokens / 1000.0 * self.output_per_1k
        )


# Conservative placeholder: priced above most small models so an unknown model
# can never be accounted as cheaper than it really is.
DEFAULT_PRICE = ModelPrice(input_per_1k=0.003, output_per_1k=0.015)

PRICES: dict[str, ModelPrice] = {
    # --- Amazon Bedrock model ids ---
    "anthropic.claude-3-5-haiku-20241022-v1:0": ModelPrice(0.0008, 0.004),
    "anthropic.claude-3-5-sonnet-20241022-v2:0": ModelPrice(0.003, 0.015),
    "amazon.nova-lite-v1:0": ModelPrice(0.00006, 0.00024),
    "amazon.nova-pro-v1:0": ModelPrice(0.0008, 0.0032),
    "meta.llama3-1-8b-instruct-v1:0": ModelPrice(0.00022, 0.00022),
    # --- OpenAI-compatible model names ---
    "gpt-4o-mini": ModelPrice(0.00015, 0.0006),
    "gpt-4o": ModelPrice(0.0025, 0.01),
    "llama-3.1-8b-instant": ModelPrice(0.00005, 0.00008),
}


def price_for(model: str) -> ModelPrice:
    """Return the price entry for `model`, falling back to `DEFAULT_PRICE`."""
    return PRICES.get(model, DEFAULT_PRICE)


def cost_usd(model: str, input_tokens: int, output_tokens: int) -> float:
    """Cost of a single call in USD."""
    return price_for(model).cost_usd(input_tokens, output_tokens)


def estimate_input_tokens(text: str) -> int:
    """Cheap pre-flight token estimate.
    """
    if not text:
        return 1
    non_ascii = sum(1 for ch in text if ord(ch) > 127)
    ratio = 2.0 if non_ascii / len(text) > 0.25 else 4.0
    return max(1, int(len(text) / ratio) + 1)
