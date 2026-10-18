"""Amazon Bedrock provider (the default).

Uses the Converse API so one code path covers Claude, Nova and Llama models
without per-family request shaping, and Titan Embeddings v2 for cache vectors.
"""

from __future__ import annotations

import json
import os

from .base import ChatRequest, ChatResponse, Provider, ProviderError

DEFAULT_EMBED_MODEL = "amazon.titan-embed-text-v2:0"


class BedrockProvider(Provider):
    name = "bedrock"

    def __init__(self, client=None, embed_model: str | None = None) -> None:
        self._client = client
        self.embed_model = embed_model or os.environ.get(
            "BEDROCK_EMBED_MODEL", DEFAULT_EMBED_MODEL
        )

    @property
    def client(self):
        # Lazily created so importing the module never requires credentials —
        # which is what lets the test suite and CI run with no AWS account.
        if self._client is None:
            import boto3

            self._client = boto3.client(
                "bedrock-runtime", region_name=os.environ.get("AWS_REGION", "us-east-1")
            )
        return self._client

    def chat(self, request: ChatRequest) -> ChatResponse:
        kwargs: dict = {
            "modelId": request.model,
            "messages": [
                {"role": m["role"], "content": [{"text": m["content"]}]}
                for m in request.messages
            ],
            "inferenceConfig": {
                "maxTokens": request.max_tokens,
                "temperature": request.temperature,
            },
        }
        if request.system:
            kwargs["system"] = [{"text": request.system}]

        try:
            raw = self.client.converse(**kwargs)
        except Exception as exc:  # noqa: BLE001 - surfaced to the caller as 502
            raise ProviderError(f"bedrock converse failed: {exc}") from exc

        try:
            text = raw["output"]["message"]["content"][0]["text"]
            usage = raw["usage"]
        except (KeyError, IndexError) as exc:
            raise ProviderError(f"unexpected bedrock response shape: {exc}") from exc

        return ChatResponse(
            text=text,
            model=request.model,
            input_tokens=int(usage["inputTokens"]),
            output_tokens=int(usage["outputTokens"]),
            provider=self.name,
            raw=raw,
        )

    def embed(self, text: str) -> list[float]:
        try:
            raw = self.client.invoke_model(
                modelId=self.embed_model,
                body=json.dumps({"inputText": text}),
                contentType="application/json",
            )
            body = json.loads(raw["body"].read())
            return [float(x) for x in body["embedding"]]
        except Exception as exc:  # noqa: BLE001
            raise ProviderError(f"bedrock embed failed: {exc}") from exc
