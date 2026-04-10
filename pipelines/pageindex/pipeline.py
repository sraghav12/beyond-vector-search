import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any, Optional

from tenacity import retry, stop_after_attempt, wait_exponential

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

try:
    from langchain_google_genai import ChatGoogleGenerativeAI
except ImportError:
    ChatGoogleGenerativeAI = None

try:
    from langchain_anthropic import ChatAnthropic
except ImportError:
    ChatAnthropic = None

try:
    from diskcache import Cache
except ImportError:
    Cache = None

from pipelines.base import (
    BasePipeline,
    CONTEXT_LIMITS,
    PipelineResult,
    TokenCounter,
    compute_cost,
)

SELECTION_SYSTEM_PROMPT = (
    "You are selecting the most relevant SEC 10-K sections from a hierarchical document outline. "
    "Return only valid JSON."
)

ANSWER_SYSTEM_PROMPT = (
    "You are a financial analyst answering questions over SEC 10-K filings. "
    "Use only the provided sections. If they are insufficient, say so clearly."
)

SECTION_HEADING_RE = re.compile(
    r"^(PART\s+[IVXLC]+|ITEM\s+\d+[A-Z]?\.?\s*.*)$",
    re.IGNORECASE,
)


def _flatten_doc(raw: dict[str, Any]) -> dict[str, Any]:
    meta = raw.get("metadata") or {}
    return {
        "doc_id": raw.get("doc_id") or meta.get("doc_id") or "UNKNOWN_DOC",
        "text": raw.get("text", ""),
        "company": raw.get("company") or meta.get("company", ""),
        "ticker": raw.get("ticker") or meta.get("ticker", ""),
        "sector": raw.get("sector") or meta.get("sector", ""),
        "fiscal_year": raw.get("fiscal_year") or meta.get("fiscal_year"),
    }


def _compact(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


class PageIndexPipeline(BasePipeline):
    name = "pageindex"

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        max_selected_sections: int = 12,
        outline_prefilter_k: int = 200,
        min_section_tokens: int = 80,
        cache_dir: str = ".cache/pageindex",
    ):
        self.model = model
        self.max_selected_sections = max_selected_sections
        self.outline_prefilter_k = outline_prefilter_k
        self.min_section_tokens = min_section_tokens

        self._cache = Cache(cache_dir) if Cache is not None else None
        self._token_counter = TokenCounter(model)
        self._context_limit = CONTEXT_LIMITS.get(model, 128_000)

        self._corpus_scale = 0
        self._docs: list[dict[str, Any]] = []
        self._sections: list[dict[str, Any]] = []
        self._sections_by_id: dict[str, dict[str, Any]] = {}
        self._outline_text = ""
        self._outline_tokens = 0

        self._llm = self._build_llm()

    def _build_llm(self):
        if self.model.startswith("gemini"):
            if ChatGoogleGenerativeAI is None:
                raise ImportError(
                    "langchain-google-genai is required for Gemini PageIndex pipeline. "
                    "Install it with: pip install langchain-google-genai"
                )
            return ChatGoogleGenerativeAI(
                model=self.model,
                temperature=0.0,
                google_api_key=os.environ["GEMINI_API_KEY"],
            )

        if self.model.startswith("claude"):
            if ChatAnthropic is None:
                raise ImportError(
                    "langchain-anthropic is required for Claude PageIndex pipeline. "
                    "Install it with: pip install langchain-anthropic"
                )
            return ChatAnthropic(
                model=self.model,
                temperature=0.0,
                api_key=os.environ["ANTHROPIC_API_KEY"],
            )

        return ChatOpenAI(
            model=self.model,
            temperature=0.0,
            api_key=os.environ["OPENAI_API_KEY"],
        )

    def _cache_key(self, question: str) -> str:
        payload = {
            "pipeline": self.name,
            "model": self.model,
            "scale": self._corpus_scale,
            "question": question,
            "max_selected_sections": self.max_selected_sections,
            "min_section_tokens": self.min_section_tokens,
        }
        raw = json.dumps(payload, sort_keys=True)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def _extract_text(self, response: Any) -> str:
        content = getattr(response, "content", response)
        if isinstance(content, str):
            return content.strip()
        if isinstance(content, list):
            parts = []
            for part in content:
                if isinstance(part, str):
                    parts.append(part)
                elif isinstance(part, dict):
                    text = part.get("text") or part.get("content")
                    if text:
                        parts.append(text)
                else:
                    text = getattr(part, "text", None)
                    if text:
                        parts.append(text)
            return "\n".join(parts).strip()
        return str(content).strip()

    def _get_usage_value(self, obj: Any, *keys: str) -> Optional[int]:
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

    def _extract_usage(self, response: Any, prompt: str, answer: str) -> tuple[int, int]:
        usage_candidates = [
            getattr(response, "usage_metadata", None),
            getattr(response, "response_metadata", {}).get("token_usage")
            if isinstance(getattr(response, "response_metadata", None), dict)
            else None,
            getattr(response, "response_metadata", {}).get("usage")
            if isinstance(getattr(response, "response_metadata", None), dict)
            else None,
        ]

        for usage in usage_candidates:
            tokens_in = self._get_usage_value(
                usage,
                "input_tokens",
                "prompt_tokens",
                "total_input_tokens",
            )
            tokens_out = self._get_usage_value(
                usage,
                "output_tokens",
                "completion_tokens",
                "total_output_tokens",
            )
            if tokens_in is not None or tokens_out is not None:
                return tokens_in or 0, tokens_out or 0

        return (
            self._token_counter.count(prompt),
            self._token_counter.count(answer),
        )

    def _read_corpus(self, corpus_path: str) -> list[dict[str, Any]]:
        docs = []
        with open(corpus_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    docs.append(_flatten_doc(json.loads(line)))
        return docs

    def _split_doc_into_sections(self, doc: dict[str, Any]) -> list[dict[str, Any]]:
        lines = doc["text"].splitlines()

        sections = []
        current_title = "Preamble"
        current_lines: list[str] = []

        def flush_section() -> None:
            nonlocal current_title, current_lines
            text = _compact("\n".join(current_lines))
            if not text:
                return
            section_id = f"{doc['doc_id']}::sec_{len(sections):03d}"
            # Skip the heading line itself for the summary so we surface actual content
            body_start = text.find(" ", len(current_title)) if current_title in text else 0
            summary_text = text[body_start:].lstrip() if body_start > 0 else text
            summary = summary_text[:220]
            sections.append(
                {
                    "section_id": section_id,
                    "doc_id": doc["doc_id"],
                    "company": doc["company"],
                    "ticker": doc["ticker"],
                    "sector": doc["sector"],
                    "fiscal_year": doc["fiscal_year"],
                    "title": current_title,
                    "summary": summary,
                    "text": text,
                }
            )

        for raw_line in lines:
            line = raw_line.strip()
            if not line:
                if current_lines and current_lines[-1] != "":
                    current_lines.append("")
                continue

            if SECTION_HEADING_RE.match(line):
                flush_section()
                current_title = line
                current_lines = [line]
            else:
                current_lines.append(line)

        flush_section()

        if not sections:
            text = _compact(doc["text"])
            if text:
                sections.append(
                    {
                        "section_id": f"{doc['doc_id']}::sec_000",
                        "doc_id": doc["doc_id"],
                        "company": doc["company"],
                        "ticker": doc["ticker"],
                        "sector": doc["sector"],
                        "fiscal_year": doc["fiscal_year"],
                        "title": "Document",
                        "summary": text[:220],
                        "text": text,
                    }
                )

        return sections

    def _build_outline_line(self, section: dict[str, Any]) -> str:
        return (
            f"[{section['section_id']}] "
            f"{section['doc_id']} | {section.get('company', '')} | "
            f"FY{section.get('fiscal_year', '')} | {section['title']} | "
            f"{section['summary']}"
        )

    def load_corpus(self, corpus_path: str) -> None:
        path = Path(corpus_path)
        try:
            self._corpus_scale = int(path.stem.split("_")[-1])
        except ValueError:
            self._corpus_scale = 0

        self._docs = self._read_corpus(str(path))
        self._sections = []
        for doc in self._docs:
            self._sections.extend(self._split_doc_into_sections(doc))

        self._sections_by_id = {section["section_id"]: section for section in self._sections}
        # Exclude stub sections (TOC entries, cross-references) from the outline so the
        # LLM never sees empty stubs competing with real content sections.
        outline_sections = [
            s for s in self._sections
            if self._token_counter.count(s["text"]) >= self.min_section_tokens
        ]
        self._outline_text = "\n".join(self._build_outline_line(s) for s in outline_sections)
        self._outline_tokens = self._token_counter.count(self._outline_text)

    def _detect_query_tickers(self, question: str) -> set[str]:
        """Return ticker symbols whose company or ticker name appears in the question."""
        q_lower = question.lower()
        ticker_to_names: dict[str, set[str]] = {}
        for section in self._sections:
            ticker = section.get("ticker", "").upper()
            company = section.get("company", "")
            if not ticker:
                continue
            if ticker not in ticker_to_names:
                ticker_to_names[ticker] = set()
            if company:
                ticker_to_names[ticker].add(company.lower())
                first_word = company.lower().split()[0]
                if len(first_word) > 3:
                    ticker_to_names[ticker].add(first_word)

        mentioned: set[str] = set()
        for ticker, names in ticker_to_names.items():
            if re.search(r"\b" + re.escape(ticker.lower()) + r"\b", q_lower):
                mentioned.add(ticker)
            for name in names:
                if name in q_lower:
                    mentioned.add(ticker)
        return mentioned

    def _lexical_prefilter_sections(self, question: str, k: int) -> list[dict[str, Any]]:
        terms = set(re.findall(r"[a-zA-Z0-9]{3,}", question.lower()))
        query_tickers = self._detect_query_tickers(question)
        scored = []

        for section in self._sections:
            if self._token_counter.count(section["text"]) < self.min_section_tokens:
                continue
            ticker = section.get("ticker", "").upper()
            title = section.get("title", "")
            haystack = " ".join(
                [
                    section["doc_id"],
                    section.get("company", ""),
                    ticker,
                    title,
                    section["summary"],
                ]
            ).lower()
            score = sum(term in haystack for term in terms)
            if score <= 0:
                continue

            # Strongly boost sections from specifically-mentioned companies so
            # financial-term overlap can't pull in the wrong company's sections.
            if query_tickers and ticker in query_tickers:
                score += 30

            # Preamble sections (cover page / TOC) rarely contain answer data.
            if title.strip().lower() == "preamble":
                score = max(0, score - 25)

            if score > 0:
                scored.append((score, section))

        if not scored:
            return self._sections[:k]

        scored.sort(key=lambda x: x[0], reverse=True)
        return [section for _, section in scored[:k]]

    def _selection_outline_for_question(self, question: str) -> tuple[str, bool, set[str]]:
        query_tickers = self._detect_query_tickers(question)
        overhead = 4_000
        if self._outline_tokens + overhead <= self._context_limit:
            return self._outline_text, True, query_tickers

        prefiltered = self._lexical_prefilter_sections(question, self.outline_prefilter_k)
        outline = "\n".join(self._build_outline_line(section) for section in prefiltered)
        return outline, False, query_tickers

    @retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=2, min=5, max=60))
    def _invoke(self, system_prompt: str, user_prompt: str):
        return self._llm.invoke(
            [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt),
            ]
        )

    def _parse_selected_section_ids(self, raw_text: str) -> list[str]:
        raw_text = raw_text.strip()

        try:
            obj = json.loads(raw_text)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", raw_text, re.DOTALL)
            if not match:
                return re.findall(r"[A-Z0-9_]+_10K_\d{4}::sec_\d{3}", raw_text)
            obj = json.loads(match.group(0))

        if isinstance(obj, dict):
            if "section_ids" in obj and isinstance(obj["section_ids"], list):
                return [str(x) for x in obj["section_ids"]]
            if "sections" in obj and isinstance(obj["sections"], list):
                values = []
                for item in obj["sections"]:
                    if isinstance(item, str):
                        values.append(item)
                    elif isinstance(item, dict) and item.get("section_id"):
                        values.append(str(item["section_id"]))
                return values

        if isinstance(obj, list):
            values = []
            for item in obj:
                if isinstance(item, str):
                    values.append(item)
                elif isinstance(item, dict) and item.get("section_id"):
                    values.append(str(item["section_id"]))
            return values

        return []

    def _fallback_section_selection(self, question: str) -> list[str]:
        return [
            section["section_id"]
            for section in self._lexical_prefilter_sections(question, self.max_selected_sections)
        ]

    def query(self, question: str) -> PipelineResult:
        if not self._sections:
            raise RuntimeError("Corpus not loaded. Call load_corpus() first.")

        cache_key = self._cache_key(question)
        if self._cache is not None and cache_key in self._cache:
            return PipelineResult(**self._cache[cache_key])

        outline_text, used_full_outline, query_tickers = self._selection_outline_for_question(question)

        company_constraint = ""
        if query_tickers:
            ticker_patterns = ", ".join(f"{t}_10K_*" for t in sorted(query_tickers))
            company_constraint = (
                f"\n- IMPORTANT: The question asks about {', '.join(sorted(query_tickers))}. "
                f"Select sections ONLY from documents matching: {ticker_patterns}. "
                f"Do NOT include sections from any other company."
            )

        selection_prompt = f"""
Question:
{question}

Outline:
{outline_text}

Return strict JSON with this schema:
{{
  "section_ids": ["DOC_ID::sec_000", "DOC_ID::sec_001"]
}}

Rules:
- Return at most {self.max_selected_sections} section_ids.
- Prefer sections likely to contain exact numeric evidence (ITEM 7, ITEM 8, financial statements).
- Avoid "Preamble" sections (cover page / table of contents) — they rarely contain answer data.
- Include multiple documents if the question is cross-document.{company_constraint}
- Return JSON only.
""".strip()

        selection_response = self._invoke(SELECTION_SYSTEM_PROMPT, selection_prompt)
        selection_text = self._extract_text(selection_response)
        selected_ids = self._parse_selected_section_ids(selection_text)

        if not selected_ids:
            selected_ids = self._fallback_section_selection(question)

        selected_sections = []
        for section_id in selected_ids:
            section = self._sections_by_id.get(section_id)
            if section is not None:
                selected_sections.append(section)

        selected_sections = selected_sections[: self.max_selected_sections]

        # Build context blocks within context window budget.
        # Reserve tokens for question, system prompt, and the answer.
        # Use 15% of context limit as reserve to account for tokenizer differences
        # between tiktoken (used for counting) and the model's actual tokenizer.
        _ANSWER_RESERVE = max(8_000, int(self._context_limit * 0.15))
        context_budget = self._context_limit - _ANSWER_RESERVE - self._token_counter.count(question)
        context_blocks = []
        used_tokens = 0
        for section in selected_sections:
            block = (
                f"[{section['section_id']}] {section['doc_id']} | "
                f"{section.get('company', '')} | FY{section.get('fiscal_year', '')}\n"
                f"Title: {section['title']}\n"
                f"{section['text']}"
            )
            block_tokens = self._token_counter.count(block)
            if used_tokens + block_tokens > context_budget:
                break
            context_blocks.append(block)
            used_tokens += block_tokens

        answer_prompt = f"""
Question:
{question}

Selected sections:
{chr(10).join(context_blocks)}

Answer using only the selected sections above.
""".strip()

        answer_response = self._invoke(ANSWER_SYSTEM_PROMPT, answer_prompt)
        answer = self._extract_text(answer_response)

        sel_in, sel_out = self._extract_usage(selection_response, selection_prompt, selection_text)
        ans_in, ans_out = self._extract_usage(answer_response, answer_prompt, answer)

        total_in = sel_in + ans_in
        total_out = sel_out + ans_out
        total_cost = compute_cost(self.model, total_in, total_out)

        result = PipelineResult(
            answer=answer,
            tokens_in=total_in,
            tokens_out=total_out,
            cost_usd=total_cost,
            corpus_scale=self._corpus_scale,
            trace={
                "question": question,
                "outline_tokens": self._outline_tokens,
                "used_full_outline": used_full_outline,
                "selected_section_ids": [s["section_id"] for s in selected_sections],
                "selected_sections": [
                    {
                        "section_id": s["section_id"],
                        "doc_id": s["doc_id"],
                        "title": s["title"],
                        "company": s.get("company"),
                        "fiscal_year": s.get("fiscal_year"),
                    }
                    for s in selected_sections
                ],
                "selection_tokens_in": sel_in,
                "selection_tokens_out": sel_out,
                "answer_tokens_in": ans_in,
                "answer_tokens_out": ans_out,
            },
        )

        if self._cache is not None:
            self._cache[cache_key] = result.__dict__

        return result
