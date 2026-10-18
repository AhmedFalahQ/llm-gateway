"""Lambda entry point.

Request lifecycle, in the order that matters:

    authenticate -> cache lookup -> reserve budget -> call provider
                 -> reconcile budget -> store in cache -> emit metrics

Two ordering decisions are load-bearing:

*Cache before budget. A hit costs nothing upstream, so it should not need
  headroom — a key that has spent its limit still gets served what the gateway
  already knows.
* Reconcile in finally. The reservation charges a worst-case estimate up
  front. If the provider call raises, the estimate must be refunded, or every
  failed request permanently eats budget.

Written against the raw API Gateway proxy event rather than a web framework:
nothing here needs routing middleware, and the dependency-free bundle keeps cold
starts short.
"""

from __future__ import annotations

import json
import time

from .budget import BudgetExceeded, BudgetStore, UnknownApiKey
from .cache import SemanticCache
from .pricing import cost_usd, estimate_input_tokens, price_for
from .providers import ChatRequest, ProviderError, get_provider
from .telemetry import emit

_provider = None
_budget: BudgetStore | None = None
_cache: SemanticCache | None = None


def _components():
    """Build the provider, budget store and cache once per container."""
    global _provider, _budget, _cache
    if _provider is None:
        _provider = get_provider()
    if _budget is None:
        _budget = BudgetStore()
    if _cache is None:
        _cache = SemanticCache(embed_fn=_provider.embed)
    return _provider, _budget, _cache


def reset_components() -> None:
    """Drop the cached singletons. Tests use this; production never calls it."""
    global _provider, _budget, _cache
    _provider = _budget = _cache = None


def set_components(provider=None, budget=None, cache=None) -> None:
    """Inject collaborators, so tests exercise the handler with no AWS calls."""
    global _provider, _budget, _cache
    _provider, _budget, _cache = provider, budget, cache


def _response(status: int, body: dict) -> dict:
    return {
        "statusCode": status,
        "headers": {"Content-Type": "application/json; charset=utf-8"},
        "body": json.dumps(body, ensure_ascii=False),
    }


def _extract_api_key(event: dict) -> str:
    headers = {k.lower(): v for k, v in (event.get("headers") or {}).items()}
    auth = headers.get("authorization", "")
    if auth.lower().startswith("bearer "):
        return auth[7:].strip()
    return headers.get("x-api-key", "").strip()


def _parse_request(body: dict) -> ChatRequest:
    messages = body.get("messages")
    if not isinstance(messages, list) or not messages:
        raise ValueError("'messages' must be a non-empty list")
    for m in messages:
        if not isinstance(m, dict) or "role" not in m or "content" not in m:
            raise ValueError("each message needs 'role' and 'content'")
    model = body.get("model")
    if not model:
        raise ValueError("'model' is required")
    return ChatRequest(
        model=model,
        messages=messages,
        max_tokens=int(body.get("max_tokens", 512)),
        temperature=float(body.get("temperature", 0.0)),
        system=body.get("system"),
    )


def _route(event: dict) -> tuple[str, str]:
    method = (
        event.get("httpMethod")
        or (event.get("requestContext", {}).get("http", {}) or {}).get("method")
        or "GET"
    ).upper()
    path = event.get("rawPath") or event.get("path") or "/"
    return method, path


def handler(event: dict, context=None) -> dict:  # noqa: ARG001 - Lambda signature
    method, path = _route(event)

    if path.endswith("/health"):
        return _response(200, {"ok": True})

    provider, budget, cache = _components()

    api_key = _extract_api_key(event)
    if not api_key:
        return _response(401, {"error": "missing api key"})
    try:
        key = budget.lookup_key(api_key)
    except UnknownApiKey as exc:
        return _response(401, {"error": str(exc)})

    if method == "GET" and path.endswith("/usage"):
        return _response(200, budget.usage(key.key_id))

    if not (method == "POST" and path.endswith("/chat")):
        return _response(404, {"error": f"no route for {method} {path}"})

    try:
        body = json.loads(event.get("body") or "{}")
        request = _parse_request(body)
    except (json.JSONDecodeError, ValueError, TypeError) as exc:
        return _response(400, {"error": f"bad request: {exc}"})

    started = time.perf_counter()

    # --- tier 1 + 2: can we answer without paying anyone? -------------------
    try:
        hit = cache.get(provider.name, request)
    except Exception:  # noqa: BLE001 - a broken cache must not break the gateway
        hit = None

    if hit is not None:
        budget.record_hit(key.key_id)
        usage = budget.usage(key.key_id)
        emit(
            key_id=key.key_id,
            model=request.model,
            cost_usd=0.0,
            latency_ms=(time.perf_counter() - started) * 1000,
            input_tokens=hit.input_tokens,
            output_tokens=hit.output_tokens,
            cache_tier=hit.tier,
        )
        return _response(
            200,
            {
                "text": hit.text,
                "model": hit.model,
                "provider": provider.name,
                "cached": True,
                "cache_tier": hit.tier,
                "similarity": hit.similarity,
                "cost_usd": 0.0,
                "saved_usd": round(
                    cost_usd(request.model, hit.input_tokens, hit.output_tokens), 6
                ),
                "usage": {
                    "input_tokens": hit.input_tokens,
                    "output_tokens": hit.output_tokens,
                },
                "budget": {
                    "limit_usd": key.monthly_limit_usd,
                    "spend_usd": usage["spend_usd"],
                    "remaining_usd": round(
                        max(0.0, key.monthly_limit_usd - usage["spend_usd"]), 6
                    ),
                },
            },
        )

    # --- admission control: charge the worst case before spending anything ---
    worst_case = price_for(request.model).cost_usd(
        estimate_input_tokens(request.prompt_text()), request.max_tokens
    )
    try:
        reservation = budget.reserve(key, worst_case)
    except BudgetExceeded as exc:
        return _response(
            429,
            {
                "error": "monthly budget exhausted",
                "limit_usd": exc.limit_usd,
                "spend_usd": round(exc.spend_usd, 6),
            },
        )

    actual = 0.0
    status = 200
    response_body: dict = {}
    try:
        result = provider.chat(request)
        actual = cost_usd(result.model, result.input_tokens, result.output_tokens)
        response_body = {
            "text": result.text,
            "model": result.model,
            "provider": result.provider,
            "cached": False,
            "cache_tier": "miss",
            "cost_usd": round(actual, 6),
            "usage": {
                "input_tokens": result.input_tokens,
                "output_tokens": result.output_tokens,
            },
        }
    except ProviderError as exc:
        status = exc.status
        response_body = {"error": str(exc)}
    finally:
        # Always settle up: `actual` is still 0.0 on the failure path, which
        # refunds the whole reservation.
        spend_after = budget.reconcile(
            reservation,
            actual,
            input_tokens=response_body.get("usage", {}).get("input_tokens", 0),
            output_tokens=response_body.get("usage", {}).get("output_tokens", 0),
        )

    if status == 200:
        try:
            cache.put(
                provider.name,
                request,
                response_body["text"],
                response_body["usage"]["input_tokens"],
                response_body["usage"]["output_tokens"],
            )
        except Exception:  # noqa: BLE001 - a failed write is a future miss, not an error
            pass

        response_body["budget"] = {
            "limit_usd": key.monthly_limit_usd,
            "spend_usd": round(spend_after, 6),
            "remaining_usd": round(max(0.0, key.monthly_limit_usd - spend_after), 6),
        }

    emit(
        key_id=key.key_id,
        model=request.model,
        cost_usd=actual,
        latency_ms=(time.perf_counter() - started) * 1000,
        input_tokens=response_body.get("usage", {}).get("input_tokens", 0),
        output_tokens=response_body.get("usage", {}).get("output_tokens", 0),
        status=status,
    )
    return _response(status, response_body)
