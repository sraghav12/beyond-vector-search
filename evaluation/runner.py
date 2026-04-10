import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from evaluation.metrics import compute_match_metrics


def load_queries(queries_path: str) -> list[dict[str, Any]]:
    with open(queries_path, encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, list):
        raise ValueError(f"Expected a list of queries in {queries_path}")
    return data


def load_gold_answers(gold_answers_path: Optional[str]) -> dict[str, dict[str, Any]]:
    if not gold_answers_path:
        return {}
    with open(gold_answers_path, encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Expected a dict of gold answers in {gold_answers_path}")
    return data


def _normalize_tier_selector(value: Any) -> str:
    value = str(value).strip().lower()
    if value.isdigit():
        return f"t{value}"
    return value


def select_queries(
    queries: list[dict[str, Any]],
    query_ids: Optional[list[str]] = None,
    tiers: Optional[list[str]] = None,
    limit: Optional[int] = None,
) -> list[dict[str, Any]]:
    query_id_set = set(query_ids or [])
    tier_set = {_normalize_tier_selector(value) for value in (tiers or [])}

    selected = []
    for query in queries:
        query_id = query.get("id", "")
        tier = query.get("tier")
        tier_name = str(query.get("tier_name", "")).strip().lower()
        tier_aliases = {
            _normalize_tier_selector(tier),
            tier_name,
            f"t{tier}" if tier is not None else "",
        }

        if query_id_set and query_id not in query_id_set:
            continue
        if tier_set and not (tier_aliases & tier_set):
            continue
        selected.append(query)

    if limit is not None:
        return selected[:limit]
    return selected


def _resolve_gold_entry(
    query: dict[str, Any],
    gold_answers: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    gold_entry = gold_answers.get(query.get("id", ""), {})
    return {
        "answer": gold_entry.get("answer", query.get("gold_answer", "")),
        "answer_type": gold_entry.get("answer_type", query.get("answer_type")),
        "difficulty": gold_entry.get("difficulty", query.get("difficulty")),
        "tier": gold_entry.get("tier", query.get("tier")),
        "tier_name": gold_entry.get("tier_name", query.get("tier_name")),
        "evidence_docs": gold_entry.get("evidence_docs", query.get("evidence_docs", [])),
        "verified": gold_entry.get("verified"),
    }


_RETRIABLE_ANSWERS = (
    "tokenlimitexceedederror",
    "cancellationerror",
    "exceeds_context",
    "error:",
)


def load_completed(filepath: str) -> set[str]:
    """Return query IDs with real answers in an output JSONL (for resume support).

    Records whose answer is a known error token (TOKENLIMITEXCEEDEDERROR, etc.)
    are excluded so they get re-run with better parameters.
    """
    path = Path(filepath)
    if not path.exists():
        return set()
    completed: set[str] = set()
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                qid = record.get("query_id", "")
                if not qid:
                    continue
                if record.get("status") == "error":
                    continue
                answer = (record.get("answer") or "").strip().lower()
                if any(answer.startswith(tok) for tok in _RETRIABLE_ANSWERS):
                    continue
                completed.add(qid)
            except json.JSONDecodeError:
                pass
    return completed


def append_jsonl(output_path: str, record: dict[str, Any]) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def build_pipeline(pipeline_name: str, **kwargs):
    if pipeline_name == "naive_llm":
        from pipelines.naive_llm.pipeline import NaiveLLMPipeline

        return NaiveLLMPipeline(**kwargs)
    if pipeline_name == "vector_rag":
        from pipelines.vector_rag.pipeline import VectorRAGPipeline

        return VectorRAGPipeline(**kwargs)
    if pipeline_name == "pageindex":
        from pipelines.pageindex.pipeline import PageIndexPipeline

        return PageIndexPipeline(**kwargs)
    if pipeline_name == "rlm":
        from pipelines.rlm.pipeline import RLMPipeline

        return RLMPipeline(**kwargs)
    raise ValueError(f"Unsupported pipeline: {pipeline_name}")


def run_pipeline(
    pipeline,
    *,
    corpus_path: str,
    queries: list[dict[str, Any]],
    gold_answers: Optional[dict[str, dict[str, Any]]] = None,
    output_path: Optional[str] = None,
    run_id: Optional[str] = None,
    resume: bool = True,
    show_progress: bool = False,
    query_timeout: Optional[int] = 120,  # seconds; None = no timeout
) -> list[dict[str, Any]]:
    try:
        from tqdm import tqdm  # type: ignore[import-untyped]
    except ImportError:
        tqdm = None  # type: ignore[assignment]

    gold_answers = gold_answers or {}
    run_id = run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    records: list[dict[str, Any]] = []

    completed_ids: set[str] = set()
    if resume and output_path:
        completed_ids = load_completed(output_path)

    load_error: Optional[Exception] = None
    try:
        pipeline.load_corpus(corpus_path)
    except Exception as exc:
        load_error = exc

    pipeline_label = getattr(pipeline, "name", pipeline.__class__.__name__)
    model_label = getattr(pipeline, "model", "")
    desc = f"{pipeline_label}/{model_label}" if model_label else pipeline_label

    query_iter: Any = queries
    if show_progress and tqdm is not None:
        query_iter = tqdm(queries, desc=desc, unit="q")

    for query in query_iter:
        if query.get("id", "") in completed_ids:
            continue
        gold_entry = _resolve_gold_entry(query, gold_answers)
        record = {
            "run_id": run_id,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "pipeline_name": getattr(pipeline, "name", pipeline.__class__.__name__),
            "model": getattr(pipeline, "model", ""),
            "corpus_path": corpus_path,
            "query_id": query.get("id", ""),
            "query_text": query.get("text", ""),
            "tier": query.get("tier"),
            "tier_name": query.get("tier_name"),
            "difficulty": gold_entry.get("difficulty"),
            "answer_type": gold_entry.get("answer_type"),
            "gold_answer": gold_entry.get("answer"),
            "evidence_docs": gold_entry.get("evidence_docs", []),
        }

        if load_error is not None:
            record.update(
                {
                    "status": "error",
                    "error_type": type(load_error).__name__,
                    "error_message": str(load_error),
                }
            )
        else:
            try:
                if query_timeout:
                    with ThreadPoolExecutor(max_workers=1) as _pool:
                        _future = _pool.submit(
                            pipeline.timed_query,
                            query["text"],
                            query_id=query.get("id", ""),
                        )
                        try:
                            result = _future.result(timeout=query_timeout)
                        except FuturesTimeoutError:
                            _future.cancel()
                            raise TimeoutError(
                                f"query exceeded {query_timeout}s wall-clock limit"
                            )
                else:
                    result = pipeline.timed_query(
                        query["text"],
                        query_id=query.get("id", ""),
                    )
                metrics = compute_match_metrics(
                    result.answer,
                    gold_entry.get("answer", ""),
                    answer_type=gold_entry.get("answer_type"),
                )
                record.update(
                    {
                        "status": "ok",
                        "answer": result.answer,
                        "latency_ms": result.latency_ms,
                        "tokens_in": result.tokens_in,
                        "tokens_out": result.tokens_out,
                        "cost_usd": result.cost_usd,
                        "corpus_scale": result.corpus_scale,
                        "trace": result.trace,
                        "metrics": metrics,
                    }
                )
            except Exception as exc:
                record.update(
                    {
                        "status": "error",
                        "error_type": type(exc).__name__,
                        "error_message": str(exc),
                    }
                )

        if output_path:
            append_jsonl(output_path, record)
        records.append(record)

    return records


def run_benchmark(
    *,
    pipeline_specs: list[dict[str, Any]],
    corpus_path: str,
    queries_path: str,
    gold_answers_path: Optional[str] = None,
    output_path: Optional[str] = None,
    query_ids: Optional[list[str]] = None,
    tiers: Optional[list[str]] = None,
    limit: Optional[int] = None,
) -> list[dict[str, Any]]:
    queries = load_queries(queries_path)
    selected_queries = select_queries(
        queries,
        query_ids=query_ids,
        tiers=tiers,
        limit=limit,
    )
    gold_answers = load_gold_answers(gold_answers_path)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    records: list[dict[str, Any]] = []
    for spec in pipeline_specs:
        pipeline_name = spec["name"]
        pipeline_kwargs = spec.get("kwargs", {})
        pipeline = build_pipeline(pipeline_name, **pipeline_kwargs)
        records.extend(
            run_pipeline(
                pipeline,
                corpus_path=corpus_path,
                queries=selected_queries,
                gold_answers=gold_answers,
                output_path=output_path,
                run_id=run_id,
            )
        )

    return records


# ── CLI ───────────────────────────────────────────────────────────────────────

_ALL_PIPELINES = ["naive_llm", "vector_rag", "pageindex", "rlm"]
_ALL_SCALES = [10, 25, 50, 100, 150]
_ALL_MODELS = ["gpt-4o-mini", "gemini-2.5-flash", "claude-haiku-4-5-20251001", "claude-sonnet-4-6"]

_DEFAULT_QUERIES = "data/queries/queries.json"
_DEFAULT_GOLD = "data/ground_truth/gold_answers.json"
_DEFAULT_OUTPUT_DIR = "results/raw"


def _parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m evaluation.runner",
        description="Run the beyond-vector-search benchmark.",
    )
    parser.add_argument("--pipeline", help="Comma-separated pipeline names, or 'all'")
    parser.add_argument("--scale", help="Comma-separated corpus scales, or 'all'")
    parser.add_argument("--model", help="Comma-separated model names, or 'all'")
    parser.add_argument("--all", dest="run_all", action="store_true",
                        help="Full sweep: all pipelines × scales × models")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print what would run without calling any APIs")
    parser.add_argument("--tier", help="Filter queries by tier (e.g. t1,t2)")
    parser.add_argument("--query-id", help="Comma-separated query IDs to run")
    parser.add_argument("--limit", type=int, help="Max queries per combination")
    parser.add_argument("--no-resume", action="store_true",
                        help="Re-run all queries even if output file exists")
    parser.add_argument("--queries", default=_DEFAULT_QUERIES)
    parser.add_argument("--gold", default=_DEFAULT_GOLD)
    parser.add_argument("--output-dir", default=_DEFAULT_OUTPUT_DIR)
    return parser.parse_args(argv)


def _resolve_list(raw: Optional[str], all_values: list) -> list:
    if not raw or raw.strip().lower() == "all":
        return list(all_values)
    return [v.strip() for v in raw.split(",") if v.strip()]


def main(argv: Optional[list[str]] = None) -> None:
    args = _parse_args(argv)

    if args.run_all:
        pipelines = _ALL_PIPELINES
        scales = _ALL_SCALES
        models = _ALL_MODELS
    else:
        pipelines = _resolve_list(args.pipeline, _ALL_PIPELINES)
        scales = [int(s) for s in _resolve_list(args.scale, _ALL_SCALES)]
        models = _resolve_list(args.model, _ALL_MODELS)

    tiers = [t.strip() for t in args.tier.split(",")] if args.tier else None
    query_ids = [q.strip() for q in args.query_id.split(",")] if args.query_id else None
    resume = not args.no_resume

    combos = [(p, m, s) for m in models for s in scales for p in pipelines]

    if args.dry_run:
        print(f"Dry run — {len(combos)} combination(s):")
        for p, m, s in combos:
            corpus = f"data/processed/corpus_{s}.jsonl"
            out = f"{args.output_dir}/{p}_{m}_{s}.jsonl"
            print(f"  {p:12s}  model={m:20s}  scale={s:4d}  corpus={corpus}  out={out}")
        return

    all_records: list[dict[str, Any]] = []
    total_cost = 0.0
    total_queries = 0
    failures = 0

    for pipeline_name, model, scale in combos:
        corpus_path = f"data/processed/corpus_{scale}.jsonl"
        output_path = f"{args.output_dir}/{pipeline_name}_{model}_{scale}.jsonl"

        pipeline = build_pipeline(pipeline_name, model=model)
        queries = load_queries(args.queries)
        selected = select_queries(
            queries,
            query_ids=query_ids,
            tiers=tiers,
            limit=args.limit,
        )
        gold = load_gold_answers(args.gold if Path(args.gold).exists() else None)

        records = run_pipeline(
            pipeline,
            corpus_path=corpus_path,
            queries=selected,
            gold_answers=gold,
            output_path=output_path,
            resume=resume,
            show_progress=True,
        )
        all_records.extend(records)
        for r in records:
            total_queries += 1
            cost = r.get("cost_usd") or 0.0
            total_cost += cost
            if r.get("status") == "error":
                failures += 1

    latencies = [
        r["latency_ms"] for r in all_records
        if r.get("latency_ms") is not None and r.get("status") == "ok"
    ]
    avg_latency = sum(latencies) / len(latencies) if latencies else 0.0

    print(
        f"\nDone. queries={total_queries}  failures={failures}"
        f"  total_cost=${total_cost:.4f}  avg_latency={avg_latency:.0f}ms"
    )


if __name__ == "__main__":
    main(sys.argv[1:])
