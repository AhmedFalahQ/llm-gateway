"""Local development provider it is only for testing purposes

     يا ربي يا حبيبي
"""

from __future__ import annotations

import hashlib
import math
import re
from .base import ChatRequest,ChatResponse,Provider

EMBED_DIM = 64
_TOKEN= re.compile(r"\w+", re.UNICODE)

def _hash_dim(token:str,dim:int)-> int:
    digest = hashlib.blake2b(token.encode("utf-8"),digest_size=8).digest()
    return int.from_bytes(digest,"big") % dim

def hashing_embed(text: str, dim: int = EMBED_DIM) -> list[float]:
    """Deterministic bag-of-words embedding via the hashing trick."""
    vector = [0.0] * dim
    for token in _TOKEN.findall(text.lower()):
        vector[_hash_dim(token, dim)] += 1.0
    norm = math.sqrt(sum(x * x for x in vector))
    if norm == 0.0:
        return [1.0] + [0.0] * (dim - 1)
    return [x / norm for x in vector]

class EchoProvider(Provider):
    name = "echo"

    def __init__(self, latency_s: float = 0.0) -> None:
        self.latency_s = latency_s
        self.calls = 0

    def chat(self, request: ChatRequest) -> ChatResponse:
        if self.latency_s:
            import time

            time.sleep(self.latency_s)
        self.calls += 1
        last = request.messages[-1]["content"]
        text = f"[echo:{request.model}] {last}"
        # Token counts are made up but proportional, so cost arithmetic and the
        # savings report produce meaningful relative numbers offline.
        return ChatResponse(
            text=text,
            model=request.model,
            input_tokens=max(1, len(request.prompt_text()) // 4),
            output_tokens=max(1, len(text) // 4),
            provider=self.name,
        )

    def embed(self, text: str) -> list[float]:
        return hashing_embed(text)