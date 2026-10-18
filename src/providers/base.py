""" The provider contract"""

from __future__ import annotations
import abc
from dataclasses import dataclass, field

@dataclass(frozen=True)
class ChatRequest:
    " Normalized chat request"
    model:str
    messages: list[dict[str,str]]
    max_tokens:int=512
    temperature:float=0.0
    system:str | None =None

    def prompt_text(self)-> str:
        "Flattened text for token est. and cache keys"
        parts=[self.system or ""]
        parts+=[f"{m.get('role','')}:{m.get('content','')}" for m in self.messages]
        return "\n".join(p for p in parts if p)

@dataclass(frozen=True)
class ChatResponse:
    """Normalized chat resp. + usage the caller is billed for"""
    text:str
    model:str
    input_tokens:int
    output_tokens:int
    provider:str
    raw: dict = field(default_factory=dict,repr=False)

class ProviderError(RuntimeError):
    """Raised when an upstream provider fails in a way the caller should see."""

    def __init__(self, message: str, status: int = 502) -> None:
        super().__init__(message)
        self.status = status
class Provider(abc.ABC):
    "Backend can answer a Chat req"
    name:str="abstract"

    @abc.abstractmethod
    def chat(self,text:ChatRequest)-> ChatResponse:
        """Send a req. upstream and return an normlized resp."""

    @abc.abstractmethod
    def embed(self,text:str)-> list[float]:
        """return an embedding for text used by semantic cache"""