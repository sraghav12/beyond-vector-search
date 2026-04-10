import json
import os
from pathlib import Path
from typing import Optional

import openai
from tenacity import retry, stop_after_attempt, wait_exponential

try:
    from google import genai as google_genai
    from google.genai import types as google_genai_types
except ImportError:
    google_genai = None
    google_genai_types = None

try:
    import google.generativeai as google_generativeai
except ImportError:
    google_generativeai = None

try:
    import anthropic as anthropic_sdk
except ImportError:
    anthropic_sdk = None

from pipelines.base import (
    BasePipeline,
    PipelineResult,
    TokenCounter,
    compute_cost,
    CONTEXT_LIMITS,
)

SYSTEM_PROMPT = (
    "You are a financial analyst. Answer the question based solely on the provided "
    "SEC 10-K filings. Be precise and cite specific numbers where relevant."
)


def _flatten_doc(raw: dict) -> dict:
    meta = raw.get("metadata") or {}
    return {
        "doc_id": raw.get("doc_id") or meta.get("doc_id") or "UNKNOWN_DOC",
        "text": raw.get("text", ""),
        "company": raw.get("company") or meta.get("company", ""),
        "fiscal_year": raw.get("fiscal_year") or meta.get("fiscal_year", ""),
    }


class NaiveLLMPipeline(BasePipeline):
    """
    Naive baseline: concatenate all corpus documents into a single LLM context
    and issue one call. If corpus exceeds the model context window, returns
    EXCEEDS_CONTEXT instead of silently truncating.
    """

    name = "naive_llm"

    def __init__(self, model: str = "gpt-4o-mini"):
        self.model = model
        self._corpus_text: str = ""
        self._corpus_tokens: int = 0
        self._corpus_scale: int = 0
        self._token_counter = TokenCounter(model)
        self._context_limit = CONTEXT_LIMITS.get(model, 128_000)
        self._is_gemini = model.startswith("gemini")
        self._is_claude = model.startswith("claude")
        self._gemini_backend: Optional[str] = None

        if self._is_gemini:
            if google_genai is not None:
                self._gemini_backend = "google_genai"
                self._gemini_client = google_genai.Client(
                    api_key=os.environ["GEMINI_API_KEY"]
                )
            elif google_generativeai is not None:
                self._gemini_backend = "google_generativeai"
                google_generativeai.configure(api_key=os.environ["GEMINI_API_KEY"])
                self._gemini_model = google_generativeai.GenerativeModel(model)
            else:
                raise ImportError(
                    "Gemini support requires either `google-genai` or "
                    "`google-generativeai` to be installed."
                )
        elif self._is_claude:
            if anthropic_sdk is None:
                raise ImportError(
                    "anthropic is required for Claude naive_llm pipeline. "
                    "Install it with: pip install anthropic"
                )
            self._anthropic_client = anthropic_sdk.Anthropic(
                api_key=os.environ["ANTHROPIC_API_KEY"]
            )
        else:
            self._openai_client = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    def load_corpus(self, corpus_path: str) -> None:
        path = Path(corpus_path)
        # Infer scale from filename, e.g. corpus_25.jsonl -> 25
        stem = path.stem  # "corpus_25"
        try:
            self._corpus_scale = int(stem.split("_")[-1])
        except ValueError:
            self._corpus_scale = 0

        docs = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    docs.append(_flatten_doc(json.loads(line)))

        parts = []
        for doc in docs:
            header = f"=== {doc.get('doc_id', 'UNKNOWN')} | {doc.get('company', '')} | FY{doc.get('fiscal_year', '')} ==="
            parts.append(f"{header}\n{doc.get('text', '')}")

        self._corpus_text = "\n\n".join(parts)
        self._corpus_tokens = self._token_counter.count(self._corpus_text)

    def query(self, question: str) -> PipelineResult:
        # Reserve ~2K tokens for system prompt + question + output
        OVERHEAD = 2_048
        if self._corpus_tokens + OVERHEAD > self._context_limit:
            return PipelineResult(
                answer="EXCEEDS_CONTEXT",
                tokens_in=self._corpus_tokens,
                tokens_out=0,
                cost_usd=0.0,
                corpus_scale=self._corpus_scale,
                trace={
                    "reason": "corpus_tokens_exceed_context_limit",
                    "corpus_tokens": self._corpus_tokens,
                    "context_limit": self._context_limit,
                },
            )

        prompt = f"{self._corpus_text}\n\nQuestion: {question}"

        if self._is_gemini:
            return self._query_gemini(question, prompt)
        if self._is_claude:
            return self._query_claude(question, prompt)
        return self._query_openai(question, prompt)

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=30))
    def _query_openai(self, question: str, prompt: str) -> PipelineResult:
        response = self._openai_client.chat.completions.create(
            model=self.model,
            temperature=0.0,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
        )
        answer = response.choices[0].message.content.strip()
        tokens_in = response.usage.prompt_tokens
        tokens_out = response.usage.completion_tokens

        return PipelineResult(
            answer=answer,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            cost_usd=compute_cost(self.model, tokens_in, tokens_out),
            corpus_scale=self._corpus_scale,
            trace={"question": question},
        )

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=30))
    def _query_gemini(self, question: str, prompt: str) -> PipelineResult:
        full_prompt = f"{SYSTEM_PROMPT}\n\n{prompt}"

        if self._gemini_backend == "google_genai":
            config = {"temperature": 0.0}
            if google_genai_types is not None:
                config = google_genai_types.GenerateContentConfig(temperature=0.0)

            response = self._gemini_client.models.generate_content(
                model=self.model,
                contents=full_prompt,
                config=config,
            )
        else:
            response = self._gemini_model.generate_content(
                full_prompt,
                generation_config=google_generativeai.GenerationConfig(temperature=0.0),
            )

        answer = getattr(response, "text", "") or ""
        answer = answer.strip()

        # Gemini returns token counts in usage_metadata
        usage = getattr(response, "usage_metadata", None)
        tokens_in = getattr(usage, "prompt_token_count", 0)
        tokens_out = getattr(usage, "candidates_token_count", 0)

        return PipelineResult(
            answer=answer,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            cost_usd=compute_cost(self.model, tokens_in, tokens_out),
            corpus_scale=self._corpus_scale,
            trace={"question": question},
        )

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=30))
    def _query_claude(self, question: str, prompt: str) -> PipelineResult:
        response = self._anthropic_client.messages.create(
            model=self.model,
            max_tokens=2048,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}],
        )
        answer = response.content[0].text.strip()
        tokens_in = response.usage.input_tokens
        tokens_out = response.usage.output_tokens

        return PipelineResult(
            answer=answer,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            cost_usd=compute_cost(self.model, tokens_in, tokens_out),
            corpus_scale=self._corpus_scale,
            trace={"question": question},
        )
