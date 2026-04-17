import math
import time
import tiktoken
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional

try:
    from dotenv import find_dotenv, load_dotenv
except ImportError:
    find_dotenv = None
    load_dotenv = None

if load_dotenv is not None and find_dotenv is not None:
    load_dotenv(find_dotenv(usecwd=True), override=True)

MODEL_COSTS = {
    "gpt-4o-mini": (0.00015, 0.00060),
    "gpt-4o": (0.00250, 0.01000),
    "gemini-2.5-flash": (0.00030, 0.00250),
    "gemini-2.0-flash": (0.00010, 0.00040),
    "gemini-1.5-flash": (0.000075, 0.000300),
    # Anthropic Claude models (per 1K tokens)
    "claude-haiku-4-5-20251001": (0.00080, 0.00400),
    "claude-haiku-4-5": (0.00080, 0.00400),
    "claude-sonnet-4-5": (0.00300, 0.01500),
    "claude-sonnet-4-6": (0.00300, 0.01500),
    "claude-opus-4-6": (0.01500, 0.07500),
}


CONTEXT_LIMITS = {
    "gpt-4o-mini": 128_000,
    "gpt-4o": 128_000,
    "gemini-2.0-flash": 1_000_000,
    "gemini-1.5-flash": 1_000_000,
    "gemini-2.5-flash": 1_000_000,
    "claude-haiku-4-5-20251001": 200_000,
    "claude-haiku-4-5": 200_000,
    "claude-sonnet-4-5": 200_000,
    "claude-sonnet-4-6": 200_000,
    "claude-opus-4-6": 200_000,
}


def compute_cost(model: str, tokens_in: int, tokens_out: int) -> float:
    if model not in MODEL_COSTS:
        return 0.0
    in_rate, out_rate = MODEL_COSTS[model]
    return (tokens_in / 1000 * in_rate) + (tokens_out / 1000 * out_rate)


class TokenCounter:
    def __init__(self, model: str = "gpt-4o-mini"):
        try:
            self._enc = tiktoken.get_encoding("cl100k_base")
        except Exception:
            self._enc = None

    def count(self, text: str) -> int:
        try:
            if self._enc is None:
                raise ValueError("Tokenizer unavailable")
            return len(self._enc.encode(text))
        except Exception:
            if not text:
                return 0
            return math.ceil(len(text) / 3.5)


@dataclass
class PipelineResult:
    answer: str
    latency_ms: float = 0.0
    tokens_in: int = 0
    tokens_out: int = 0
    cost_usd: float = 0.0
    pipeline_name: str = ""
    model: str = ""
    query_id: str = ""
    corpus_scale: int = 0
    trace: Optional[dict] = field(default=None)


class BasePipeline(ABC):
    name: str = "base"
    model: str = ""

    @abstractmethod
    def load_corpus(self, corpus_path: str) -> None: ...

    @abstractmethod
    def query(self, question: str) -> PipelineResult: ...

    def timed_query(self, question: str, query_id: str = "") -> PipelineResult:
        start = time.perf_counter()
        result = self.query(question)
        result.latency_ms = (time.perf_counter() - start) * 1000
        result.pipeline_name = self.name
        result.model = self.model
        result.query_id = query_id
        return result
