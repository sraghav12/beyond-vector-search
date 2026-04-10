import json
import math
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

import pipelines.base as base_module
import pipelines.vector_rag.pipeline as vector_rag_module


def _write_corpus(path: Path, docs: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for doc in docs:
            handle.write(json.dumps(doc) + "\n")


class FakeCache(dict):
    def __init__(self, directory: str):
        super().__init__()
        self.directory = directory


class FakeResponse:
    def __init__(self, content: str, token_usage: dict[str, int]):
        self.content = content
        self.response_metadata = {"token_usage": token_usage}


class FakeLLM:
    def __init__(
        self,
        content: str = "Stub answer",
        token_usage: dict[str, int] | None = None,
    ):
        self.content = content
        self.token_usage = token_usage or {
            "prompt_tokens": 17,
            "completion_tokens": 9,
        }
        self.calls = 0
        self.messages = []

    def invoke(self, messages):
        self.calls += 1
        self.messages.append(messages)
        return FakeResponse(self.content, self.token_usage)


class FakeChroma:
    def __init__(
        self,
        collection_name: str,
        persist_directory: str,
        embedding_function,
    ):
        self.collection_name = collection_name
        self.persist_directory = Path(persist_directory)
        self.embedding_function = embedding_function
        self.docs = []
        self.ids = []
        self.add_calls = 0
        self.search_calls = 0

        self.persist_directory.mkdir(parents=True, exist_ok=True)
        (self.persist_directory / "chroma.sqlite3").write_text(
            "initialized",
            encoding="utf-8",
        )

    def add_documents(self, documents, ids):
        self.add_calls += 1
        self.docs = list(documents)
        self.ids = list(ids)

    def similarity_search_with_relevance_scores(self, question: str, k: int):
        self.search_calls += 1
        return [
            (doc, 0.95 - (idx * 0.1))
            for idx, doc in enumerate(self.docs[:k])
        ]


def _make_pipeline(
    monkeypatch,
    tmp_path: Path,
    *,
    llm_content: str = "Stub answer",
    llm_usage: dict[str, int] | None = None,
    **kwargs,
):
    llm = FakeLLM(content=llm_content, token_usage=llm_usage)

    monkeypatch.setattr(vector_rag_module, "Cache", FakeCache)
    monkeypatch.setattr(vector_rag_module, "Chroma", FakeChroma)
    monkeypatch.setattr(
        vector_rag_module.VectorRAGPipeline,
        "_build_embeddings",
        lambda self: object(),
    )
    monkeypatch.setattr(
        vector_rag_module.VectorRAGPipeline,
        "_build_llm",
        lambda self: llm,
    )

    params = {
        "persist_root": str(tmp_path / "chroma"),
        "cache_dir": str(tmp_path / "cache"),
        "chunk_size": 2048,
        "chunk_overlap": 0,
    }
    params.update(kwargs)

    pipeline = vector_rag_module.VectorRAGPipeline(**params)
    return pipeline, llm


def test_load_corpus_builds_new_index_even_when_chroma_creates_files(
    monkeypatch,
    tmp_path,
):
    corpus_path = tmp_path / "corpus_10.jsonl"
    _write_corpus(
        corpus_path,
        [
            {
                "doc_id": "AAPL_10K_2025",
                "company": "Apple",
                "ticker": "AAPL",
                "sector": "Technology",
                "fiscal_year": 2025,
                "filing_date": "2026-01-30",
                "text": "Revenue increased to 100 units.",
            },
            {
                "doc_id": "MSFT_10K_2025",
                "company": "Microsoft",
                "ticker": "MSFT",
                "sector": "Technology",
                "fiscal_year": 2025,
                "filing_date": "2026-02-01",
                "text": "Operating income increased to 50 units.",
            },
        ],
    )

    pipeline, _ = _make_pipeline(monkeypatch, tmp_path)
    pipeline.load_corpus(str(corpus_path))

    assert pipeline._vectorstore.add_calls == 1
    assert pipeline._vectorstore.ids == [
        chunk.metadata["chunk_id"] for chunk in pipeline._chunks
    ]
    assert pipeline._index_embedding_tokens > 0
    assert pipeline._index_embedding_cost_usd > 0


def test_load_corpus_reuses_existing_index_without_reembedding(
    monkeypatch,
    tmp_path,
):
    corpus_path = tmp_path / "corpus_10.jsonl"
    _write_corpus(
        corpus_path,
        [
            {
                "doc_id": "AAPL_10K_2025",
                "company": "Apple",
                "ticker": "AAPL",
                "sector": "Technology",
                "fiscal_year": 2025,
                "filing_date": "2026-01-30",
                "text": "Revenue increased to 100 units.",
            }
        ],
    )

    persist_root = tmp_path / "chroma"
    existing_index_dir = (
        persist_root
        / "gpt-4o-mini"
        / "text-embedding-3-small"
        / "scale_10"
    )
    existing_index_dir.mkdir(parents=True, exist_ok=True)
    (existing_index_dir / "chroma.sqlite3").write_text("existing", encoding="utf-8")

    pipeline, _ = _make_pipeline(
        monkeypatch,
        tmp_path,
        persist_root=str(persist_root),
    )
    pipeline.load_corpus(str(corpus_path))

    assert pipeline._vectorstore.add_calls == 0
    assert pipeline._index_embedding_tokens == 0
    assert pipeline._index_embedding_cost_usd == 0.0


def test_query_returns_answer_and_uses_cache(monkeypatch, tmp_path):
    corpus_path = tmp_path / "corpus_10.jsonl"
    _write_corpus(
        corpus_path,
        [
            {
                "doc_id": "AAPL_10K_2025",
                "company": "Apple",
                "ticker": "AAPL",
                "sector": "Technology",
                "fiscal_year": 2025,
                "filing_date": "2026-01-30",
                "text": "Revenue increased to 100 units.",
            },
            {
                "doc_id": "MSFT_10K_2025",
                "company": "Microsoft",
                "ticker": "MSFT",
                "sector": "Technology",
                "fiscal_year": 2025,
                "filing_date": "2026-02-01",
                "text": "Operating income increased to 50 units.",
            },
        ],
    )

    pipeline, llm = _make_pipeline(
        monkeypatch,
        tmp_path,
        llm_content="Revenue was 100 units.",
        llm_usage={"prompt_tokens": 21, "completion_tokens": 6},
        top_k=2,
    )
    pipeline.load_corpus(str(corpus_path))

    first = pipeline.query("What was Apple's revenue?")
    second = pipeline.query("What was Apple's revenue?")

    assert first.answer == "Revenue was 100 units."
    assert first.tokens_in == (
        first.trace["generation_tokens_in"] + first.trace["query_embedding_tokens"]
    )
    assert first.tokens_out == 6
    assert first.trace["retrieved_chunks"]
    assert all(
        item["included_in_prompt"] for item in first.trace["retrieved_chunks"]
    )
    assert pipeline._vectorstore.search_calls == 1
    assert llm.calls == 1
    assert second.answer == first.answer
    assert second.trace == first.trace


def test_query_marks_trimmed_context_when_some_chunks_are_omitted(
    monkeypatch,
    tmp_path,
):
    corpus_path = tmp_path / "corpus_10.jsonl"
    _write_corpus(
        corpus_path,
        [
            {
                "doc_id": "AAPL_10K_2025",
                "company": "Apple",
                "ticker": "AAPL",
                "sector": "Technology",
                "fiscal_year": 2025,
                "filing_date": "2026-01-30",
                "text": "Alpha " * 20,
            },
            {
                "doc_id": "MSFT_10K_2025",
                "company": "Microsoft",
                "ticker": "MSFT",
                "sector": "Technology",
                "fiscal_year": 2025,
                "filing_date": "2026-02-01",
                "text": "Beta " * 20,
            },
        ],
    )

    pipeline, llm = _make_pipeline(
        monkeypatch,
        tmp_path,
        llm_content="Trimmed answer",
        top_k=2,
    )
    pipeline.load_corpus(str(corpus_path))
    pipeline._token_counter.count = lambda text: len(text)
    pipeline._PROMPT_TOKEN_OVERHEAD = 0

    question = "Compare the two filings."
    first_block = pipeline._build_context_block(pipeline._vectorstore.docs[0])
    prompt_prefix = f"Question: {question}\n\nRetrieved context:\n\n"
    prompt_suffix = "\n\nAnswer the question using only this retrieved context."
    pipeline._context_limit = (
        len(vector_rag_module.SYSTEM_PROMPT)
        + len(prompt_prefix)
        + len(first_block)
        + len(prompt_suffix)
    )

    result = pipeline.query(question)

    assert result.answer == "Trimmed answer"
    assert result.trace["context_trimmed"] is True
    assert len(result.trace["omitted_chunk_ids"]) == 1
    assert result.trace["retrieved_chunks"][0]["included_in_prompt"] is True
    assert result.trace["retrieved_chunks"][1]["included_in_prompt"] is False
    assert llm.calls == 1


def test_query_returns_exceeds_context_when_no_chunk_fits(monkeypatch, tmp_path):
    corpus_path = tmp_path / "corpus_10.jsonl"
    _write_corpus(
        corpus_path,
        [
            {
                "doc_id": "AAPL_10K_2025",
                "company": "Apple",
                "ticker": "AAPL",
                "sector": "Technology",
                "fiscal_year": 2025,
                "filing_date": "2026-01-30",
                "text": "Gamma " * 30,
            }
        ],
    )

    pipeline, llm = _make_pipeline(monkeypatch, tmp_path, top_k=1)
    pipeline.load_corpus(str(corpus_path))
    pipeline._token_counter.count = lambda text: len(text)
    pipeline._PROMPT_TOKEN_OVERHEAD = 0

    question = "What does the filing say?"
    first_block = pipeline._build_context_block(pipeline._vectorstore.docs[0])
    prompt_prefix = f"Question: {question}\n\nRetrieved context:\n\n"
    prompt_suffix = "\n\nAnswer the question using only this retrieved context."
    pipeline._context_limit = (
        len(vector_rag_module.SYSTEM_PROMPT)
        + len(prompt_prefix)
        + len(prompt_suffix)
        + len(first_block)
        - 1
    )

    result = pipeline.query(question)

    assert result.answer == "EXCEEDS_CONTEXT"
    assert result.tokens_out == 0
    assert result.trace["reason"] == "retrieved_context_exceeds_context_limit"
    assert result.trace["retrieved_chunks"][0]["included_in_prompt"] is False
    assert llm.calls == 0


def test_token_counter_fallback_uses_character_ratio(monkeypatch):
    def raise_on_get_encoding(_name: str):
        raise RuntimeError("offline")

    monkeypatch.setattr(base_module.tiktoken, "get_encoding", raise_on_get_encoding)

    counter = base_module.TokenCounter()

    assert counter.count("") == 0
    assert counter.count("abcdefgh") == math.ceil(8 / 3.5)
