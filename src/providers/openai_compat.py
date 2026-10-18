"""OpenAI-compatible provider.

Covers OpenAI itself and anything that speaks the same two endpoints — Groq,
Together, OpenRouter, vLLM, Ollama. Implemented with `urllib` rather than the
`openai` SDK to keep the Lambda bundle at zero extra megabytes.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request

from .base import ChatRequest, ChatResponse, Provider, ProviderError

DEFAULT_BASE_URL = "https://api.openai.com/v1"
DEFAULT_EMBED_MODEL = "text-embedding-3-small"


class OpenAICompatProvider(Provider):
    name = "openai-compat"

    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        embed_model: str | None = None,
        timeout: float = 30.0,
        opener=None,
    ) -> None:
        self.base_url = (base_url or os.environ.get("OPENAI_BASE_URL", DEFAULT_BASE_URL)).rstrip(
            "/"
        )
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self.embed_model = embed_model or os.environ.get("OPENAI_EMBED_MODEL", DEFAULT_EMBED_MODEL)
        self.timeout = timeout
        # Injectable so tests exercise the real request-building code without a
        # network round trip.
        self._opener = opener or urllib.request.urlopen

    def _post(self, path: str, payload: dict) -> dict:
        req = urllib.request.Request(
            f"{self.base_url}{path}",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )
        try:
            with self._opener(req, timeout=self.timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", "replace")[:500]
            raise ProviderError(f"upstream {exc.code}: {detail}", status=502) from exc
        except Exception as exc:  # noqa: BLE001
            raise ProviderError(f"upstream request failed: {exc}") from exc

    def chat(self, request: ChatRequest) -> ChatResponse:
        messages = list(request.messages)
        if request.system:
            messages = [{"role": "system", "content": request.system}, *messages]

        raw = self._post(
            "/chat/completions",
            {
                "model": request.model,
                "messages": messages,
                "max_tokens": request.max_tokens,
                "temperature": request.temperature,
            },
        )

        try:
            text = raw["choices"][0]["message"]["content"]
            usage = raw.get("usage", {})
        except (KeyError, IndexError) as exc:
            raise ProviderError(f"unexpected response shape: {exc}") from exc

        return ChatResponse(
            text=text,
            model=request.model,
            input_tokens=int(usage.get("prompt_tokens", 0)),
            output_tokens=int(usage.get("completion_tokens", 0)),
            provider=self.name,
            raw=raw,
        )

    def embed(self, text: str) -> list[float]:
        raw = self._post("/embeddings", {"model": self.embed_model, "input": text})
        try:
            return [float(x) for x in raw["data"][0]["embedding"]]
        except (KeyError, IndexError) as exc:
            raise ProviderError(f"unexpected embedding shape: {exc}") from exc
