"""Provider registry."""

from __future__ import annotations

import os

from .base import ChatRequest, ChatResponse, Provider, ProviderError
from .bedrock import BedrockProvider
from .echo import EchoProvider
from .openai_compat import OpenAICompatProvider

_REGISTRY: dict[str, type[Provider]] = {
    "bedrock": BedrockProvider,
    "openai-compat": OpenAICompatProvider,
    "echo": EchoProvider,  # local development only; see providers/echo.py
}

__all__ = [
    "ChatRequest",
    "ChatResponse",
    "Provider",
    "ProviderError",
    "BedrockProvider",
    "EchoProvider",
    "OpenAICompatProvider",
    "get_provider",
    "register",
]


def register(name: str, cls: type[Provider]) -> None:
    """Register a provider implementation under `name`."""
    _REGISTRY[name] = cls


def get_provider(name: str | None = None) -> Provider:
    """Build the configured provider. Defaults to Bedrock."""
    key = (name or os.environ.get("LLM_PROVIDER", "bedrock")).lower()
    try:
        return _REGISTRY[key]()
    except KeyError:
        raise ProviderError(
            f"unknown provider {key!r}; known: {sorted(_REGISTRY)}", status=400
        ) from None
