import json
from pathlib import Path

from evaluation.runner import run_benchmark, run_pipeline, select_queries
from pipelines.base import BasePipeline, PipelineResult


FIXTURES = Path(__file__).parent / "fixtures"


class FakePipeline(BasePipeline):
    name = "fake_pipeline"

    def __init__(self, answer: str = "81,000", fail_query_id: str | None = None):
        self.model = "fake-model"
        self.answer = answer
        self.fail_query_id = fail_query_id
        self.loaded_corpus_path = None
        self.active_query_id = None

    def load_corpus(self, corpus_path: str) -> None:
        self.loaded_corpus_path = corpus_path

    def query(self, question: str) -> PipelineResult:
        if self.active_query_id == self.fail_query_id:
            raise RuntimeError("simulated failure")
        return PipelineResult(
            answer=self.answer,
            tokens_in=12,
            tokens_out=4,
            cost_usd=0.123,
            corpus_scale=10,
            trace={"question": question},
        )

    def timed_query(self, question: str, query_id: str = "") -> PipelineResult:
        self.active_query_id = query_id
        return super().timed_query(question, query_id=query_id)


def test_select_queries_supports_ids_tiers_and_limit():
    queries = [
        {"id": "q1", "tier": 1, "tier_name": "single_hop"},
        {"id": "q2", "tier": 2, "tier_name": "multi_hop"},
        {"id": "q3", "tier": 1, "tier_name": "single_hop"},
    ]

    selected = select_queries(queries, query_ids=["q1", "q3"], tiers=["t1"], limit=1)

    assert selected == [{"id": "q1", "tier": 1, "tier_name": "single_hop"}]


def test_run_pipeline_appends_jsonl_and_computes_metrics(tmp_path):
    queries = json.loads((FIXTURES / "sample_queries.json").read_text(encoding="utf-8"))
    gold_answers = json.loads(
        (FIXTURES / "sample_gold_answers.json").read_text(encoding="utf-8")
    )
    output_path = tmp_path / "results.jsonl"

    pipeline = FakePipeline(answer="81,000")
    records = run_pipeline(
        pipeline,
        corpus_path=str(FIXTURES / "sample_corpus.jsonl"),
        queries=queries[:1],
        gold_answers=gold_answers,
        output_path=str(output_path),
        run_id="smoke_run",
    )

    assert len(records) == 1
    assert pipeline.loaded_corpus_path.endswith("sample_corpus.jsonl")
    assert records[0]["status"] == "ok"
    assert records[0]["metrics"]["strict_match"] is False
    assert records[0]["metrics"]["lenient_match"] is True

    written_lines = output_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(written_lines) == 1
    written_record = json.loads(written_lines[0])
    assert written_record["pipeline_name"] == "fake_pipeline"
    assert written_record["query_id"] == "q_smoke_001"


def test_run_benchmark_uses_pipeline_specs_and_continues_after_query_errors(
    monkeypatch,
    tmp_path,
):
    output_path = tmp_path / "benchmark_runs.jsonl"

    def fake_build_pipeline(pipeline_name: str, **kwargs):
        assert pipeline_name == "vector_rag"
        return FakePipeline(answer="81,000", fail_query_id="q_smoke_002")

    monkeypatch.setattr("evaluation.runner.build_pipeline", fake_build_pipeline)

    records = run_benchmark(
        pipeline_specs=[{"name": "vector_rag", "kwargs": {"model": "fake"}}],
        corpus_path=str(FIXTURES / "sample_corpus.jsonl"),
        queries_path=str(FIXTURES / "sample_queries.json"),
        gold_answers_path=str(FIXTURES / "sample_gold_answers.json"),
        output_path=str(output_path),
        limit=2,
    )

    assert len(records) == 2
    assert records[0]["status"] == "ok"
    assert records[1]["status"] == "error"
    assert records[1]["error_type"] == "RuntimeError"

    written_lines = output_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(written_lines) == 2
