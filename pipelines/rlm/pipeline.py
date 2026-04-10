import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any, Optional

from tenacity import RetryError, retry, retry_if_not_exception_type, stop_after_attempt, wait_exponential

try:
    from diskcache import Cache
except ImportError:
    Cache = None

try:
    from rlm import RLM
except ImportError:
    RLM = None

try:
    from rlm.logger import RLMLogger
except ImportError:
    RLMLogger = None

try:
    from rlm.utils.exceptions import (
        BudgetExceededError,
        CancellationError,
        ErrorThresholdExceededError,
        TimeoutExceededError,
        TokenLimitExceededError,
    )
except ImportError:
    BudgetExceededError = None
    CancellationError = None
    ErrorThresholdExceededError = None
    TimeoutExceededError = None
    TokenLimitExceededError = None

from pipelines.base import BasePipeline, PipelineResult, TokenCounter, compute_cost

RLM_QUERY_TEMPLATE = """You are answering a question over a long SEC 10-K corpus.

The full corpus is provided separately as `context`.
Use code execution and recursive sub-calls when useful.
Search for exact figures before answering.
Be concise and evidence-grounded.

Question:
{question}
"""

NON_RETRYABLE_RLM_EXCEPTIONS = tuple(
    exc
    for exc in (
        BudgetExceededError,
        CancellationError,
        ErrorThresholdExceededError,
        TimeoutExceededError,
        TokenLimitExceededError,
    )
    if exc is not None
)


class RLMPipeline(BasePipeline):
    name = "rlm"

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        backend: Optional[str] = None,
        environment: str = "local",
        max_depth: int = 4,
        max_subcalls: int = 12,
        token_budget: int = 8000,
        log_dir: str = ".cache/rlm_logs",
        cache_dir: str = ".cache/rlm",
        verbose: bool = False,
    ):
        self.model = model
        self.backend = backend
        self.environment = environment
        self.max_depth = max_depth
        self.max_subcalls = max_subcalls
        self.token_budget = token_budget
        self.verbose = verbose

        self._cache = Cache(cache_dir) if Cache is not None else None
        self._log_dir = Path(log_dir)
        self._log_dir.mkdir(parents=True, exist_ok=True)

        self._token_counter = TokenCounter(model)
        self._corpus_scale = 0
        self._corpus_text = ""
        self._corpus_tokens = 0

        self._backend_name, self._backend_kwargs = self._build_backend_config()
        self._rlm = self._build_rlm_client()

    def _build_backend_config(self) -> tuple[str, dict[str, Any]]:
        if self.backend:
            if self.backend == "openrouter":
                return self.backend, {"model_name": f"google/{self.model}"}
            return self.backend, {"model_name": self.model}

        if self.model.startswith("gpt"):
            return "openai", {"model_name": self.model}

        if self.model.startswith("gemini"):
            if os.getenv("OPENROUTER_API_KEY"):
                return "openrouter", {"model_name": f"google/{self.model}"}
            return "litellm", {
                "model_name": f"gemini/{self.model}",
                "api_key": os.environ["GEMINI_API_KEY"],
            }

        return "litellm", {"model_name": self.model}

    def _build_rlm_client(self):
        if RLM is None:
            raise ImportError("rlms is required for RLMPipeline. Install it with: pip install rlms")

        kwargs: dict[str, Any] = {
            "backend": self._backend_name,
            "backend_kwargs": self._backend_kwargs,
            "environment": self.environment,
            "max_depth": self.max_depth,
            # The current rlms package exposes iteration and token caps rather than
            # the max_subcalls/token_budget names used in this project's config.
            "max_iterations": self.max_subcalls,
            # 0 means no token cap — execution is bounded by max_iterations/max_depth only
            "max_tokens": self.token_budget if self.token_budget > 0 else None,
            "verbose": self.verbose,
        }

        if RLMLogger is not None:
            kwargs["logger"] = RLMLogger(log_dir=str(self._log_dir))

        return RLM(**kwargs)

    def _cache_key(self, question: str) -> str:
        payload = {
            "pipeline": self.name,
            "model": self.model,
            "backend": self._backend_name,
            "scale": self._corpus_scale,
            "question": question,
            "max_depth": self.max_depth,
            "max_subcalls": self.max_subcalls,
            "token_budget": self.token_budget,
        }
        raw = json.dumps(payload, sort_keys=True)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def load_corpus(self, corpus_path: str) -> None:
        path = Path(corpus_path)
        try:
            self._corpus_scale = int(path.stem.split("_")[-1])
        except ValueError:
            self._corpus_scale = 0

        docs = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                raw = json.loads(line)
                meta = raw.get("metadata") or {}
                doc_id = raw.get("doc_id") or meta.get("doc_id") or "UNKNOWN_DOC"
                company = raw.get("company") or meta.get("company", "")
                fiscal_year = raw.get("fiscal_year") or meta.get("fiscal_year", "")
                text = raw.get("text", "")
                docs.append(f"=== {doc_id} | {company} | FY{fiscal_year} ===\n{text}")

        self._corpus_text = "\n\n".join(docs)
        self._corpus_tokens = self._token_counter.count(self._corpus_text)

    def _extract_answer(self, response: Any) -> str:
        candidate = None
        for attr in ("response", "final_answer", "answer", "text"):
            value = getattr(response, attr, None)
            if value:
                candidate = str(value).strip()
                break

        metadata = getattr(response, "metadata", None)
        if metadata:
            resolved = self._extract_answer_from_metadata(metadata, candidate)
            if resolved:
                return resolved

        if candidate:
            return candidate
        return str(response).strip()

    def _extract_answer_from_metadata(
        self,
        metadata: Any,
        candidate: Optional[str] = None,
    ) -> Optional[str]:
        if not isinstance(metadata, dict):
            return None

        candidate_is_var = bool(candidate and re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", candidate))

        for iteration in reversed(metadata.get("iterations", [])):
            final_answer = iteration.get("final_answer")
            if isinstance(final_answer, str) and final_answer.strip():
                answer = final_answer.strip()
                if not candidate_is_var or answer != candidate:
                    return answer

            for block in reversed(iteration.get("code_blocks", [])):
                result = block.get("result") or {}
                result_final_answer = result.get("final_answer")
                if isinstance(result_final_answer, str) and result_final_answer.strip():
                    answer = result_final_answer.strip()
                    if not candidate_is_var or answer != candidate:
                        return answer

                if candidate_is_var:
                    locals_dict = result.get("locals") or {}
                    if candidate in locals_dict:
                        value = locals_dict[candidate]
                        if value is not None:
                            return str(value).strip()

        return None

    def _get_value(self, obj: Any, *keys: str) -> Optional[int]:
        if obj is None:
            return None
        if isinstance(obj, dict):
            for key in keys:
                value = obj.get(key)
                if value is not None:
                    return int(value)
            return None
        for key in keys:
            value = getattr(obj, key, None)
            if value is not None:
                return int(value)
        return None

    def _extract_usage(self, response: Any, question: str, answer: str) -> tuple[int, int, bool]:
        candidates = [
            getattr(response, "usage_summary", None),
            getattr(response, "usage", None),
            getattr(response, "token_usage", None),
            getattr(response, "response_metadata", None),
        ]

        for usage in candidates:
            tokens_in = self._get_value(
                usage,
                "input_tokens",
                "prompt_tokens",
                "total_input_tokens",
                "prompt_token_count",
            )
            tokens_out = self._get_value(
                usage,
                "output_tokens",
                "completion_tokens",
                "total_output_tokens",
                "candidates_token_count",
            )
            if tokens_in is not None or tokens_out is not None:
                return tokens_in or 0, tokens_out or 0, False

        return (
            self._token_counter.count(question),
            self._token_counter.count(answer),
            True,
        )

    def _latest_trajectory_log(self) -> Optional[str]:
        if not self._log_dir.exists():
            return None
        files = sorted(self._log_dir.glob("*.jsonl"), key=lambda p: p.stat().st_mtime)
        return str(files[-1]) if files else None

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=20),
        retry=retry_if_not_exception_type((RuntimeError,) + NON_RETRYABLE_RLM_EXCEPTIONS),
    )
    def _call_rlm(self, question: str):
        root_prompt = RLM_QUERY_TEMPLATE.format(question=question)
        try:
            return self._rlm.completion(
                prompt=self._corpus_text,
                root_prompt=root_prompt,
            )
        except TypeError as exc:
            raise RuntimeError(
                "Your installed rlms version exposes a different completion() signature than "
                "the one assumed here. Update only _call_rlm(); the rest of the pipeline is ready."
            ) from exc

    def _failure_result(self, reason: str, message: str) -> PipelineResult:
        return PipelineResult(
            answer=reason,
            tokens_in=0,
            tokens_out=0,
            cost_usd=0.0,
            corpus_scale=self._corpus_scale,
            trace={
                "reason": reason,
                "error": message,
                "backend": self._backend_name,
                "backend_kwargs": {
                    k: v for k, v in self._backend_kwargs.items() if "key" not in k.lower()
                },
                "corpus_tokens": self._corpus_tokens,
                "trajectory_log": self._latest_trajectory_log(),
                "max_depth": self.max_depth,
                "max_subcalls": self.max_subcalls,
                "token_budget": self.token_budget,
            },
        )

    def query(self, question: str) -> PipelineResult:
        if not self._corpus_text:
            raise RuntimeError("Corpus not loaded. Call load_corpus() first.")

        cache_key = self._cache_key(question)
        if self._cache is not None and cache_key in self._cache:
            return PipelineResult(**self._cache[cache_key])

        try:
            response = self._call_rlm(question)
        except RetryError as exc:
            last_error = exc.last_attempt.exception()
            if isinstance(last_error, NON_RETRYABLE_RLM_EXCEPTIONS):
                return self._failure_result(
                    type(last_error).__name__.upper(),
                    str(last_error),
                )
            raise
        except NON_RETRYABLE_RLM_EXCEPTIONS as exc:
            return self._failure_result(
                type(exc).__name__.upper(),
                str(exc),
            )

        answer = self._extract_answer(response)
        tokens_in, tokens_out, estimated = self._extract_usage(response, question, answer)
        cost = compute_cost(self.model, tokens_in, tokens_out)

        result = PipelineResult(
            answer=answer,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            cost_usd=cost,
            corpus_scale=self._corpus_scale,
            trace={
                "question": question,
                "backend": self._backend_name,
                "backend_kwargs": {
                    k: v for k, v in self._backend_kwargs.items() if "key" not in k.lower()
                },
                "corpus_tokens": self._corpus_tokens,
                "trajectory_log": self._latest_trajectory_log(),
                "usage_estimated": estimated,
                "max_depth": self.max_depth,
                "max_subcalls": self.max_subcalls,
                "token_budget": self.token_budget,
            },
        )

        if self._cache is not None:
            self._cache[cache_key] = result.__dict__

        return result
