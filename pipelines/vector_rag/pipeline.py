import hashlib
import json
import os
import re
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any, Optional

from tenacity import retry, stop_after_attempt, wait_exponential

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

try:
    from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
except ImportError:
    ChatGoogleGenerativeAI = None
    GoogleGenerativeAIEmbeddings = None

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

SYSTEM_PROMPT = (
    "You are a financial analyst answering questions over SEC 10-K filings. "
    "Use only the retrieved context. If the context is insufficient, say so clearly. "
    "Prefer exact figures, dates, and concise evidence-backed answers."
)

EMBEDDING_COSTS_PER_1K = {
    "text-embedding-3-small": 0.00002,
    "text-embedding-3-large": 0.00013,
    "gemini-embedding-001": 0.00015,
    "models/gemini-embedding-001": 0.00015,
}


def _slug(value: str) -> str:
    value = re.sub(r"[^a-zA-Z0-9_-]+", "_", value.strip().lower())
    return value.strip("_") or "default"


def _flatten_doc(raw: dict[str, Any]) -> dict[str, Any]:
    meta = raw.get("metadata") or {}
    return {
        "doc_id": raw.get("doc_id") or meta.get("doc_id") or "UNKNOWN_DOC",
        "text": raw.get("text", ""),
        "company": raw.get("company") or meta.get("company", ""),
        "ticker": raw.get("ticker") or meta.get("ticker", ""),
        "sector": raw.get("sector") or meta.get("sector", ""),
        "fiscal_year": raw.get("fiscal_year") or meta.get("fiscal_year"),
        "filing_date": raw.get("filing_date") or meta.get("filing_date"),
    }


class VectorRAGPipeline(BasePipeline):
    name = "vector_rag"
    _PROMPT_TOKEN_OVERHEAD = 2_048

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        embedding_model: Optional[str] = None,
        embedding_batch_size: int = 32,
        chunk_size: int = 1024,
        chunk_overlap: int = 128,
        top_k: int = 5,
        persist_root: str = ".cache/chroma",
        cache_dir: str = ".cache/vector_rag",
        rebuild_index: bool = False,
    ):
        self.model = model
        self.embedding_model = embedding_model or (
            "models/gemini-embedding-001"
            if model.startswith("gemini")
            else "text-embedding-3-small"
        )
        self.embedding_batch_size = embedding_batch_size
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.top_k = top_k
        self.rebuild_index = rebuild_index

        self._persist_root = Path(persist_root)
        self._cache = Cache(cache_dir) if Cache is not None else None

        self._token_counter = TokenCounter(model)
        self._context_limit = CONTEXT_LIMITS.get(model, 128_000)

        self._corpus_scale = 0
        self._docs: list[dict[str, Any]] = []
        self._chunks: list[Document] = []
        self._vectorstore: Optional[Chroma] = None

        self._index_embedding_tokens = 0
        self._index_embedding_cost_usd = 0.0

        self._embeddings = self._build_embeddings()
        self._llm = self._build_llm()

    def _build_llm(self):
        if self.model.startswith("gemini"):
            if ChatGoogleGenerativeAI is None:
                raise ImportError(
                    "langchain-google-genai is required for Gemini generation. "
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
                    "langchain-anthropic is required for Claude generation. "
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

    def _build_embeddings(self):
        if self.embedding_model.startswith("models/gemini") or self.embedding_model.startswith("gemini"):
            if GoogleGenerativeAIEmbeddings is None:
                raise ImportError(
                    "langchain-google-genai is required for Gemini embeddings. "
                    "Install it with: pip install langchain-google-genai"
                )
            return GoogleGenerativeAIEmbeddings(
                model=self.embedding_model,
                google_api_key=os.environ["GEMINI_API_KEY"],
            )

        return OpenAIEmbeddings(
            model=self.embedding_model,
            api_key=os.environ["OPENAI_API_KEY"],
            chunk_size=self.embedding_batch_size,
        )

    def _collection_name(self) -> str:
        base = (
            f"bvs_{self.model}_{self.embedding_model}_"
            f"s{self._corpus_scale}_c{self.chunk_size}_o{self.chunk_overlap}"
        )
        base = _slug(base)
        if len(base) <= 50:
            return base
        digest = hashlib.md5(base.encode("utf-8")).hexdigest()[:10]
        return f"{base[:39]}_{digest}"

    def _index_dir(self) -> Path:
        model_slug = _slug(self.model)
        embed_slug = _slug(self.embedding_model)
        return self._persist_root / model_slug / embed_slug / f"scale_{self._corpus_scale}"

    def _cache_key(self, question: str) -> str:
        payload = {
            "pipeline": self.name,
            "model": self.model,
            "embedding_model": self.embedding_model,
            "embedding_batch_size": self.embedding_batch_size,
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "top_k": self.top_k,
            "scale": self._corpus_scale,
            "question": question,
        }
        raw = json.dumps(payload, sort_keys=True)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def _estimate_embedding_cost(self, tokens: int) -> float:
        rate = EMBEDDING_COSTS_PER_1K.get(self.embedding_model, 0.0)
        return (tokens / 1000.0) * rate

    def _estimate_embedding_tokens(self, texts: list[str]) -> int:
        return sum(self._token_counter.count(text) for text in texts)

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

        fallback_in = self._token_counter.count(SYSTEM_PROMPT) + self._token_counter.count(prompt)
        fallback_out = self._token_counter.count(answer)
        return fallback_in, fallback_out

    def _read_corpus(self, corpus_path: str) -> list[dict[str, Any]]:
        docs = []
        with open(corpus_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    docs.append(_flatten_doc(json.loads(line)))
        return docs

    def _chunk_documents(self, docs: list[dict[str, Any]]) -> list[Document]:
        splitter_kwargs = {
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "separators": ["\n\n", "\n", ". ", " ", ""],
        }

        try:
            splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
                encoding_name="cl100k_base",
                **splitter_kwargs,
            )
        except Exception:
            splitter = RecursiveCharacterTextSplitter(**splitter_kwargs)

        lc_docs = []
        for doc in docs:
            lc_docs.append(
                Document(
                    page_content=doc["text"],
                    metadata={
                        "doc_id": doc["doc_id"],
                        "company": doc["company"],
                        "ticker": doc["ticker"],
                        "sector": doc["sector"],
                        "fiscal_year": doc["fiscal_year"],
                        "filing_date": doc["filing_date"],
                    },
                )
            )

        split_docs = splitter.split_documents(lc_docs)

        per_doc_counter: defaultdict[str, int] = defaultdict(int)
        for chunk in split_docs:
            doc_id = chunk.metadata["doc_id"]
            idx = per_doc_counter[doc_id]
            per_doc_counter[doc_id] += 1
            chunk.metadata["chunk_index"] = idx
            chunk.metadata["chunk_id"] = f"{doc_id}::chunk_{idx:04d}"

        return split_docs

    def _index_exists(self, index_dir: Path) -> bool:
        return index_dir.exists() and any(index_dir.iterdir())

    def _build_context_block(self, doc: Document) -> str:
        return (
            f"[{doc.metadata['chunk_id']}] "
            f"{doc.metadata.get('doc_id', '')} | "
            f"{doc.metadata.get('company', '')} | "
            f"FY{doc.metadata.get('fiscal_year', '')}\n"
            f"{doc.page_content}"
        )

    def _retrieval_trace_entry(
        self,
        doc: Document,
        score: float,
        included_in_prompt: bool,
    ) -> dict[str, Any]:
        return {
            "chunk_id": doc.metadata.get("chunk_id"),
            "doc_id": doc.metadata.get("doc_id"),
            "company": doc.metadata.get("company"),
            "fiscal_year": doc.metadata.get("fiscal_year"),
            "score": float(score),
            "included_in_prompt": included_in_prompt,
        }

    def load_corpus(self, corpus_path: str) -> None:
        path = Path(corpus_path)
        try:
            self._corpus_scale = int(path.stem.split("_")[-1])
        except ValueError:
            self._corpus_scale = 0

        self._docs = self._read_corpus(str(path))
        self._chunks = self._chunk_documents(self._docs)

        index_dir = self._index_dir()
        if self.rebuild_index and index_dir.exists():
            shutil.rmtree(index_dir)
        index_exists = self._index_exists(index_dir)

        self._vectorstore = Chroma(
            collection_name=self._collection_name(),
            persist_directory=str(index_dir),
            embedding_function=self._embeddings,
        )

        if index_exists and not self.rebuild_index:
            self._index_embedding_tokens = 0
            self._index_embedding_cost_usd = 0.0
            return

        if not self._chunks:
            self._index_embedding_tokens = 0
            self._index_embedding_cost_usd = 0.0
            return

        ids = [chunk.metadata["chunk_id"] for chunk in self._chunks]
        _CHROMA_MAX_BATCH = 5000
        for i in range(0, len(self._chunks), _CHROMA_MAX_BATCH):
            self._vectorstore.add_documents(
                self._chunks[i : i + _CHROMA_MAX_BATCH],
                ids=ids[i : i + _CHROMA_MAX_BATCH],
            )

        self._index_embedding_tokens = self._estimate_embedding_tokens(
            [chunk.page_content for chunk in self._chunks]
        )
        self._index_embedding_cost_usd = self._estimate_embedding_cost(
            self._index_embedding_tokens
        )

    @retry(stop=stop_after_attempt(6), wait=wait_exponential(multiplier=2, min=5, max=90))
    def _generate_answer(self, prompt: str):
        return self._llm.invoke(
            [
                SystemMessage(content=SYSTEM_PROMPT),
                HumanMessage(content=prompt),
            ]
        )

    def query(self, question: str) -> PipelineResult:
        if self._vectorstore is None:
            raise RuntimeError("Corpus not loaded. Call load_corpus() first.")

        cache_key = self._cache_key(question)
        if self._cache is not None and cache_key in self._cache:
            return PipelineResult(**self._cache[cache_key])

        retrieved = self._vectorstore.similarity_search_with_relevance_scores(
            question,
            k=self.top_k,
        )

        question_embedding_tokens = self._estimate_embedding_tokens([question])
        question_embedding_cost = self._estimate_embedding_cost(question_embedding_tokens)

        prompt_prefix = f"Question: {question}\n\nRetrieved context:\n\n"
        prompt_suffix = "\n\nAnswer the question using only this retrieved context."

        context_blocks = []
        retrieved_trace = []
        omitted_chunk_ids = []
        context_trimmed = False
        for doc, score in retrieved:
            block = self._build_context_block(doc)
            candidate_blocks = context_blocks + [block]
            candidate_prompt = (
                prompt_prefix
                + "\n\n".join(candidate_blocks)
                + prompt_suffix
            )
            candidate_tokens = (
                self._token_counter.count(SYSTEM_PROMPT)
                + self._token_counter.count(candidate_prompt)
            )

            if candidate_tokens + self._PROMPT_TOKEN_OVERHEAD <= self._context_limit:
                context_blocks.append(block)
                retrieved_trace.append(self._retrieval_trace_entry(doc, score, True))
                continue

            context_trimmed = True
            chunk_id = doc.metadata.get("chunk_id")
            if chunk_id:
                omitted_chunk_ids.append(chunk_id)
            retrieved_trace.append(self._retrieval_trace_entry(doc, score, False))

        if retrieved and not context_blocks:
            result = PipelineResult(
                answer="EXCEEDS_CONTEXT",
                tokens_in=question_embedding_tokens,
                tokens_out=0,
                cost_usd=question_embedding_cost,
                corpus_scale=self._corpus_scale,
                trace={
                    "reason": "retrieved_context_exceeds_context_limit",
                    "question": question,
                    "query_embedding_tokens": question_embedding_tokens,
                    "query_embedding_cost_usd": question_embedding_cost,
                    "context_limit": self._context_limit,
                    "retrieved_chunks": retrieved_trace,
                    "omitted_chunk_ids": omitted_chunk_ids,
                    "chunk_size": self.chunk_size,
                    "chunk_overlap": self.chunk_overlap,
                    "top_k": self.top_k,
                    "embedding_model": self.embedding_model,
                },
            )

            if self._cache is not None:
                self._cache[cache_key] = result.__dict__

            return result

        prompt = prompt_prefix + "\n\n".join(context_blocks) + prompt_suffix
        prompt_tokens = (
            self._token_counter.count(SYSTEM_PROMPT)
            + self._token_counter.count(prompt)
        )

        response = self._generate_answer(prompt)
        answer = self._extract_text(response)
        gen_tokens_in, gen_tokens_out = self._extract_usage(response, prompt, answer)
        generation_cost = compute_cost(self.model, gen_tokens_in, gen_tokens_out)

        result = PipelineResult(
            answer=answer,
            tokens_in=gen_tokens_in + question_embedding_tokens,
            tokens_out=gen_tokens_out,
            cost_usd=generation_cost + question_embedding_cost,
            corpus_scale=self._corpus_scale,
            trace={
                "question": question,
                "retrieved_chunks": retrieved_trace,
                "query_embedding_tokens": question_embedding_tokens,
                "query_embedding_cost_usd": question_embedding_cost,
                "generation_tokens_in": gen_tokens_in,
                "generation_tokens_out": gen_tokens_out,
                "generation_cost_usd": generation_cost,
                "index_embedding_tokens": self._index_embedding_tokens,
                "index_embedding_cost_usd": self._index_embedding_cost_usd,
                "embedding_batch_size": self.embedding_batch_size,
                "chunk_size": self.chunk_size,
                "chunk_overlap": self.chunk_overlap,
                "top_k": self.top_k,
                "embedding_model": self.embedding_model,
                "prompt_tokens_estimate": prompt_tokens,
                "context_limit": self._context_limit,
                "context_trimmed": context_trimmed,
                "omitted_chunk_ids": omitted_chunk_ids,
            },
        )

        if self._cache is not None:
            self._cache[cache_key] = result.__dict__

        return result
