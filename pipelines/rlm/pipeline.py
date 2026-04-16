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

RLM_QUERY_TEMPLATE = """You are answering a question over a corpus of SEC 10-K filings.

The corpus is in the REPL as `context`: a list of strings, one per filing.
Each filing begins with `=== <DOC_ID> | <COMPANY> | FY<YEAR> ===` then the
filing body. Example headers: `=== AAPL_10K_2024 | Apple | FY2024 ===`,
`=== NVDA_10K_2025 | NVIDIA | FY2025 ===`.

CRITICAL REPL RULES:
- The REPL runs with `exec()`. It CANNOT echo last-expression values. You MUST
  call `print(...)` on anything you want to see. Writing `my_var` alone on a
  line shows NOTHING. Always wrap in `print(my_var)` or `print(repr(my_var))`.
- Do your work in ONE REPL block end-to-end: select doc, slice it, call sub-LLM,
  print the result. Don't split "do work" and "display" into separate turns.

CRITICAL SUB-LLM SIZE RULE:
- The sub-LLM (`llm_query`) has a context limit of ~128K tokens
  (~512K characters). Most 10-K filings fit. Some (JPMorgan, Pfizer, etc.)
  DO NOT.
- If `len(doc) <= 450000`: pass the WHOLE doc unsliced. Slicing a doc that
  fits only hurts recall.
- If `len(doc) > 450000`: you MUST slice. Use KEYWORD-ANCHORED slicing —
  search the doc for one of these anchor phrases (in order):
    "Consolidated Statements of Operations"
    "Consolidated Statements of Income"
    "Total revenues"
    "Total revenue"
    "Net sales"
    "Total net revenue"
  Take a window of ~200,000 chars centered on the first hit. This is the
  financial-statements section where the numbers live. Only if NO anchor
  is found, fall back to `doc[len(doc)//2 - 200000 : len(doc)//2 + 200000]`.
- If a sub-LLM call still returns a context-length error, slice smaller
  (100K chars around the anchor) and retry. NEVER give up and write prose
  about "token limits" — just slice smaller.
- If a sub-LLM returns NOT_FOUND, try a DIFFERENT anchor keyword or a
  different slice position BEFORE emitting FINAL(UNKNOWN). A 10-K always
  contains the company's total revenue; if you didn't find it, you sliced
  the wrong section.

CRITICAL FINAL ANSWER RULE:
- As soon as you have the numeric figure from the sub-LLM, your VERY NEXT
  response must be a single line of plain text: `FINAL(<the figure>)`.
- FINAL(...) must be at the START of the line with NOTHING before it — not
  "Here's the answer:", not markdown, not a code fence. Just: `FINAL($391,035 million)`
- NEVER wrap FINAL() inside a ```repl``` block. FINAL() is parsed from the
  assistant text, not executed in the REPL.
- NEVER output code as your final response. If you can't get the number,
  output exactly: `FINAL(UNKNOWN)` so the pipeline can mark it a failure
  instead of scoring your explanation.

Recommended single-block strategy:
```repl
# 1. Pick the relevant filing by header
headers = [doc.split('\\n', 1)[0] for doc in context]
print("HEADERS:", headers)
candidates = [i for i, h in enumerate(headers) if 'Apple' in h and '2024' in h]
print("candidates:", candidates)

# 2. Conditionally slice ONLY if the doc is too big
doc = context[candidates[0]]
print("doc length:", len(doc))
MAX_CHARS = 450000
if len(doc) <= MAX_CHARS:
    doc_slice = doc  # small enough — pass whole doc
else:
    # Keyword-anchored slice around the financial statements.
    anchors = [
        "Consolidated Statements of Operations",
        "Consolidated Statements of Income",
        "Total revenues",
        "Total revenue",
        "Net sales",
        "Total net revenue",
    ]
    hit = -1
    for a in anchors:
        hit = doc.find(a)
        if hit != -1:
            print("anchor hit:", repr(a), "@", hit)
            break
    if hit == -1:
        mid = len(doc) // 2
        doc_slice = doc[max(0, mid - 200000): mid + 200000]
        print("no anchor found; using center slice")
    else:
        lo = max(0, hit - 50000)
        hi = min(len(doc), hit + 350000)
        doc_slice = doc[lo:hi]
print("slice length:", len(doc_slice))

# 3. Extract the precise figure with a tight sub-LLM prompt
answer = llm_query(
    "From the SEC 10-K filing excerpt below, extract Apple's total net sales "
    "(revenue) for fiscal year 2024. Report ONLY the exact dollar figure as "
    "stated in the filing (e.g. '$391,035 million' or '$391.0 billion'). "
    "If the figure is not in this excerpt, respond exactly: NOT_FOUND. "
    "No prose, no explanation.\\n\\n" + doc_slice
)
print("ANSWER:", answer)
```
Then in your NEXT turn, on its own line, with NOTHING else:
`FINAL(<the exact figure from ANSWER>)`

If ANSWER was NOT_FOUND, try a DIFFERENT anchor keyword or expand the window
and retry BEFORE giving up. Only emit `FINAL(UNKNOWN)` after two failed
slices. Every 10-K contains its total revenue — if you didn't find it, your
slice missed the section.

Rules for the answer:
- Use the exact figure as reported. Never paraphrase or round silently.
- Match the question's fiscal year. 10-Ks show multiple years for comparison;
  take the one matching the doc header's FY, not the comparative prior-year.

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
        max_depth: int = 2,
        max_subcalls: int = 20,
        token_budget: int = 0,
        max_timeout: float = 540.0,
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
        self.max_timeout = max_timeout
        self.verbose = verbose

        self._cache = Cache(cache_dir) if Cache is not None else None
        self._log_dir = Path(log_dir)
        self._log_dir.mkdir(parents=True, exist_ok=True)

        self._token_counter = TokenCounter(model)
        self._corpus_scale = 0
        self._corpus_docs: list[str] = []
        self._corpus_text = ""
        self._corpus_tokens = 0

        self._backend_name, self._backend_kwargs = self._build_backend_config()
        self._rlm = self._build_rlm_client()

    def _build_backend_config(self) -> tuple[str, dict[str, Any]]:
        if self.backend:
            if self.backend == "openrouter":
                return self.backend, {"model_name": f"google/{self.model}"}
            if self.backend == "anthropic":
                return self.backend, {
                    "model_name": self.model,
                    "api_key": os.environ["ANTHROPIC_API_KEY"],
                }
            return self.backend, {"model_name": self.model}

        if self.model.startswith("gpt"):
            return "openai", {"model_name": self.model}

        if self.model.startswith("claude"):
            return "anthropic", {
                "model_name": self.model,
                "api_key": os.environ["ANTHROPIC_API_KEY"],
            }

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
            "max_iterations": self.max_subcalls,
            "max_tokens": self.token_budget if self.token_budget > 0 else None,
            "max_timeout": self.max_timeout if self.max_timeout and self.max_timeout > 0 else None,
            "max_errors": 3,
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
            "max_timeout": self.max_timeout,
            "context_format": "list_per_doc_v6_anchor_slice",
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

        self._corpus_docs = docs
        self._corpus_text = "\n\n".join(docs)
        self._corpus_tokens = self._token_counter.count(self._corpus_text)

    # Lenient FINAL(...) extractor. rlms only parses the strict
    # `^\s*FINAL\((.*)\)\s*$` form, but gpt-4o-mini often emits FINAL(...)
    # buried in prose or with trailing text. We grab the LAST FINAL(...)
    # in the string and return its argument.
    _FINAL_WRAPPER_RE = re.compile(r"FINAL\s*\(\s*(.*?)\s*\)\s*\.?\s*$", re.DOTALL)
    _FINAL_ANY_RE = re.compile(r"FINAL\s*\(\s*(.*?)\s*\)", re.DOTALL)
    # Heuristics for "this isn't an answer, it's leaked scratch work."
    _UNPARSEABLE_PREFIXES = (
        "```",
        "# ",
        "to resolve",
        "here's how",
        "here is how",
        "here is the adjusted",
        "here's the adjusted",
        "let's ",
        "we'll need",
        "we need to",
        "i will ",
        "i'll ",
        "the adjusted code",
        "the following code",
    )
    # Substrings that indicate a "false abstention" — the model gave up mid-
    # search and described the gap instead of emitting FINAL(UNKNOWN). These
    # match anywhere in the (stripped, lowered) answer, not just prefix.
    _FALSE_ABSTENTION_SUBSTRINGS = (
        "could not be found",
        "could not find",
        "not be found in",
        "not available in",
        "not documented in",
        "is not present in",
        "is not mentioned in",
        "was not found in",
        "not included in the provided",
        "not included in the available",
        "not in the excerpt",
        "not in the available",
    )

    def _strip_final_wrapper(self, text: str) -> str:
        """Strip one or more FINAL(...) wrappers and surrounding quotes."""
        if not text:
            return text
        s = text.strip()
        # Try anchored-at-end first (most common when rlms fellthrough cleanly).
        m = self._FINAL_WRAPPER_RE.search(s)
        if m:
            s = m.group(1).strip()
        else:
            # Fall back to last FINAL(...) anywhere in the string.
            matches = list(self._FINAL_ANY_RE.finditer(s))
            if matches:
                s = matches[-1].group(1).strip()
        # Strip balanced outer quotes.
        while len(s) >= 2 and s[0] == s[-1] and s[0] in ("'", '"'):
            s = s[1:-1].strip()
        return s

    def _looks_unparseable(self, text: str) -> bool:
        """True if text is clearly not a real answer (code/prose/strategy)."""
        if not text:
            return True
        s = text.strip()
        if not s:
            return True
        # Explicit UNKNOWN signal from the prompt's retry protocol.
        if s.upper() == "UNKNOWN":
            return True
        lowered = s.lower()
        for prefix in self._UNPARSEABLE_PREFIXES:
            if lowered.startswith(prefix):
                return True
        # False-abstention prose: model gave up instead of emitting
        # FINAL(UNKNOWN). We treat it as the same failure.
        for sub in self._FALSE_ABSTENTION_SUBSTRINGS:
            if sub in lowered:
                return True
        # Contains a REPL fence anywhere = leaked scratch.
        if "```repl" in lowered or "```python" in lowered:
            return True
        # Long prose with no dollar figure and no clear number at all = junk.
        # (Keep threshold generous so real short answers like "15%" aren't killed.)
        if len(s) > 400:
            has_dollar = bool(re.search(r"\$\s?[\d,]+(?:\.\d+)?", s))
            has_large_num = bool(re.search(r"\b\d{1,3}(?:,\d{3})+\b", s))
            if not (has_dollar or has_large_num):
                return True
        return False

    def _extract_answer(self, response: Any) -> Optional[str]:
        """
        Returns the cleaned final answer string, or None if the response
        did not contain a real answer (caller should treat as failure).
        """
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
                candidate = resolved

        if candidate is None:
            candidate = str(response).strip()

        # Post-process: strip FINAL() wrappers the strict rlms parser missed.
        cleaned = self._strip_final_wrapper(candidate)

        # Classify: is this actually an answer, or leaked scratch work?
        if self._looks_unparseable(cleaned):
            return None
        return cleaned

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
        # Pass corpus as list[str] (one element per document) so the RLM's REPL
        # sees `context` as an iterable with per-doc lengths in the metadata
        # prompt — letting it chunk by document instead of slicing one giant blob.
        try:
            return self._rlm.completion(
                prompt=self._corpus_docs,
                root_prompt=root_prompt,
            )
        except TypeError as exc:
            raise RuntimeError(
                "Your installed rlms version exposes a different completion() signature than "
                "the one assumed here. Update only _call_rlm(); the rest of the pipeline is ready."
            ) from exc

    def _failure_result(self, reason: str, message: str) -> PipelineResult:
        # NOTE: failure results are intentionally NOT cached so a single transient
        # timeout doesn't permanently poison the cache for that question.
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
                "corpus_docs": len(self._corpus_docs),
                "trajectory_log": self._latest_trajectory_log(),
                "max_depth": self.max_depth,
                "max_subcalls": self.max_subcalls,
                "token_budget": self.token_budget,
                "max_timeout": self.max_timeout,
            },
        )

    def query(self, question: str) -> PipelineResult:
        if not self._corpus_docs:
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
        if answer is None:
            # Model ran but didn't produce a real final answer (leaked code or
            # strategy prose). Don't let this reach the judge as a scored 0 —
            # surface it as a structured failure so we can count it separately.
            raw = ""
            for attr in ("response", "final_answer", "answer", "text"):
                value = getattr(response, attr, None)
                if value:
                    raw = str(value).strip()
                    break
            return self._failure_result(
                "NO_FINAL_ANSWER",
                f"RLM returned non-answer text (first 300 chars): {raw[:300]}",
            )
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
                "corpus_docs": len(self._corpus_docs),
                "trajectory_log": self._latest_trajectory_log(),
                "usage_estimated": estimated,
                "max_depth": self.max_depth,
                "max_subcalls": self.max_subcalls,
                "token_budget": self.token_budget,
                "max_timeout": self.max_timeout,
            },
        )

        if self._cache is not None:
            self._cache[cache_key] = result.__dict__

        return result
