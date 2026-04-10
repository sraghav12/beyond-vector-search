#!/usr/bin/env python3
"""Benchmark orchestrator.

Examples:
    # One pipeline, one scale — development / debug
    python scripts/run_benchmark.py --pipeline naive_llm --scale 10 --model gpt-4o-mini

    # All pipelines, all scales, one model
    python scripts/run_benchmark.py --pipeline all --scale all --model gpt-4o-mini

    # Dry run — show plan without calling APIs
    python scripts/run_benchmark.py --pipeline all --scale all --model gpt-4o-mini --dry-run

    # Judge-only — score existing result files without re-running pipelines
    python scripts/run_benchmark.py --judge-only --judge-model gpt-4o-mini
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

# ── project root on sys.path so imports work when called as a script ─────────
_ROOT = Path(__file__).parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from evaluation.judge import LLMJudge, get_recommended_judge
from evaluation.runner import (
    build_pipeline,
    load_completed,
    load_gold_answers,
    load_queries,
    run_pipeline,
    select_queries,
)

# ── constants ─────────────────────────────────────────────────────────────────

ALL_PIPELINES = ["naive_llm", "vector_rag", "pageindex", "rlm"]
ALL_SCALES = [10, 25, 50, 100, 150]
ALL_MODELS = ["gpt-4o-mini", "gemini/gemini-2.5-flash-preview-04-17"]

DEFAULT_QUERIES = "data/queries/queries.json"
DEFAULT_GOLD = "data/ground_truth/gold_answers.json"
DEFAULT_CORPUS_DIR = "data/processed"
DEFAULT_OUTPUT_DIR = "results"

log = logging.getLogger(__name__)


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="run_benchmark",
        description="Beyond-Vector-Search benchmark runner.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # -- what to run --
    run_grp = p.add_argument_group("run selection")
    run_grp.add_argument(
        "--pipeline",
        default="naive_llm",
        help="Pipeline(s): naive_llm, vector_rag, pageindex, rlm, all, or comma-separated.",
    )
    run_grp.add_argument(
        "--scale",
        default="10",
        help="Corpus scale(s): 10, 25, 50, 100, 150, all, or comma-separated.",
    )
    run_grp.add_argument(
        "--model",
        default="gpt-4o-mini",
        help="Model(s) comma-separated, or 'all'.",
    )
    run_grp.add_argument("--tier", help="Filter queries by tier(s), comma-separated.")
    run_grp.add_argument("--query-ids", help="Run specific query IDs, comma-separated.")
    run_grp.add_argument("--limit", type=int, help="Max queries per combination.")

    # -- modes --
    mode_grp = p.add_argument_group("execution mode")
    mode_grp.add_argument("--dry-run", action="store_true")
    mode_grp.add_argument("--no-judge", action="store_true", help="Skip judging step.")
    mode_grp.add_argument(
        "--judge-only",
        action="store_true",
        help="Only judge existing result files; skip pipeline runs.",
    )
    mode_grp.add_argument("--judge-model", help="Override judge model (auto-selected by default).")
    mode_grp.add_argument("--no-resume", action="store_true", help="Re-run all queries, even completed ones.")

    # -- paths --
    path_grp = p.add_argument_group("paths")
    path_grp.add_argument("--queries", default=DEFAULT_QUERIES)
    path_grp.add_argument("--corpus-dir", default=DEFAULT_CORPUS_DIR)
    path_grp.add_argument("--gold", default=DEFAULT_GOLD)
    path_grp.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)

    # -- per-pipeline tuning --
    tune_grp = p.add_argument_group("pipeline tuning")
    tune_grp.add_argument(
        "--query-timeout", type=int, default=120,
        help="Per-query wall-clock timeout in seconds. 0 = no limit. (default: 120)",
    )
    tune_grp.add_argument("--vector-top-k", type=int, default=5)
    tune_grp.add_argument("--vector-chunk-size", type=int, default=1024)
    tune_grp.add_argument("--vector-chunk-overlap", type=int, default=128)
    tune_grp.add_argument("--vector-embedding-batch-size", type=int, default=32)
    tune_grp.add_argument("--vector-rebuild-index", action="store_true")
    tune_grp.add_argument("--pageindex-max-sections", type=int, default=8)
    tune_grp.add_argument("--rlm-max-depth", type=int, default=4)
    tune_grp.add_argument("--rlm-max-subcalls", type=int, default=12)
    tune_grp.add_argument("--rlm-token-budget", type=int, default=8000)

    p.add_argument("--verbose", action="store_true")
    return p.parse_args(argv)


# ── helpers ───────────────────────────────────────────────────────────────────

def _resolve(raw: str, all_values: list) -> list:
    if raw.strip().lower() == "all":
        return list(all_values)
    return [v.strip() for v in raw.split(",") if v.strip()]


def _model_safe(model: str) -> str:
    """Filesystem-safe model name: replace slashes and other unsafe chars."""
    return model.replace("/", "_").replace(":", "_").replace(".", "-")


def _raw_path(output_dir: str, pipeline: str, model: str, scale: int) -> Path:
    return Path(output_dir) / "raw" / f"{pipeline}_{_model_safe(model)}_{scale}.jsonl"


def _metrics_path(output_dir: str, pipeline: str, model: str, scale: int) -> Path:
    return (
        Path(output_dir)
        / "metrics"
        / f"{pipeline}_{_model_safe(model)}_{scale}_scored.csv"
    )


def _pipeline_kwargs(pipeline_name: str, model: str, args: argparse.Namespace) -> dict:
    if pipeline_name == "naive_llm":
        return {"model": model}
    if pipeline_name == "vector_rag":
        return {
            "model": model,
            "top_k": args.vector_top_k,
            "chunk_size": args.vector_chunk_size,
            "chunk_overlap": args.vector_chunk_overlap,
            "embedding_batch_size": args.vector_embedding_batch_size,
            "rebuild_index": args.vector_rebuild_index,
        }
    if pipeline_name == "pageindex":
        return {"model": model, "max_selected_sections": args.pageindex_max_sections}
    if pipeline_name == "rlm":
        return {
            "model": model,
            "max_depth": args.rlm_max_depth,
            "max_subcalls": args.rlm_max_subcalls,
            "token_budget": args.rlm_token_budget,
        }
    return {"model": model}


# ── dry run ───────────────────────────────────────────────────────────────────

def dry_run(
    pipelines: list[str],
    scales: list[int],
    models: list[str],
    n_queries: int,
    judge_model: str | None,
) -> None:
    combos = [(p, m, s) for m in models for s in scales for p in pipelines]
    total_q = len(combos) * n_queries
    print("=== BENCHMARK EXECUTION PLAN ===")
    print(f"Pipelines : {', '.join(pipelines)}")
    print(f"Scales    : {', '.join(str(s) for s in scales)}")
    print(f"Models    : {', '.join(models)}")
    print(f"Queries   : {n_queries} per combination")
    if judge_model:
        print(f"Judge     : {judge_model}")
    print()
    print(f"Total runs    : {len(combos)} ({len(pipelines)} pipelines × {len(scales)} scales × {len(models)} models)")
    print(f"Total queries : {total_q}")
    print()
    print("Would execute:")
    for i, (pipeline, model, scale) in enumerate(combos, 1):
        print(f"  {i:3d}. {pipeline:12s} | scale={scale:4d} | model={model}")
    print()
    print("Use without --dry-run to execute.")


# ── judge helpers ─────────────────────────────────────────────────────────────

def run_judge(
    raw_path: Path,
    gold_path: str,
    metrics_path: Path,
    judge_model_override: str | None,
    pipeline_model: str,
) -> float:
    """Judge one results file. Returns total judge cost."""
    if not raw_path.exists():
        log.warning("Results file not found, skipping judge: %s", raw_path)
        return 0.0
    judge_model = judge_model_override or get_recommended_judge(pipeline_model)
    judge = LLMJudge(model=pipeline_model)
    # Pass judge_model directly; LLMJudge infers the cross-model inside score()
    df = judge.score_results_file(
        str(raw_path),
        gold_path,
        output_path=str(metrics_path),
    )
    _ = df  # DataFrame already saved by score_results_file
    return judge.total_cost_usd


# ── aggregation ───────────────────────────────────────────────────────────────

def aggregate_all(metrics_dir: Path, output_dir: Path) -> None:
    import pandas as pd  # type: ignore[import-untyped]

    csv_files = list(metrics_dir.glob("*_scored.csv"))
    if not csv_files:
        log.warning("No scored CSVs found in %s — skipping aggregation.", metrics_dir)
        return

    dfs = []
    for f in csv_files:
        try:
            dfs.append(pd.read_csv(f))
        except Exception as exc:
            log.warning("Could not read %s: %s", f, exc)

    if not dfs:
        return

    master = pd.concat(dfs, ignore_index=True)
    summary_path = metrics_dir / "summary.csv"
    master.to_csv(summary_path, index=False)
    print(f"\nSummary saved → {summary_path}  ({len(master)} rows)")

    # Print accuracy table by pipeline × tier
    score_col = "judge_score" if "judge_score" in master.columns else None
    if score_col and "tier_name" in master.columns and "pipeline" in master.columns:
        print("\nAccuracy by pipeline × tier (mean judge score):")
        pivot = (
            master.groupby(["pipeline", "tier_name"])[score_col]
            .mean()
            .unstack(fill_value=float("nan"))
            .round(3)
        )
        print(pivot.to_string())


def write_cost_report(cost_rows: list[dict], output_dir: Path) -> None:
    report = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "by_run": cost_rows,
        "totals": {
            "pipeline_cost_usd": round(sum(r.get("pipeline_cost_usd", 0) for r in cost_rows), 6),
            "judge_cost_usd": round(sum(r.get("judge_cost_usd", 0) for r in cost_rows), 6),
        },
    }
    path = output_dir / "cost_report.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        json.dump(report, fh, indent=2)
    total = report["totals"]["pipeline_cost_usd"] + report["totals"]["judge_cost_usd"]
    print(f"\nCost report → {path}")
    print(f"  Pipeline : ${report['totals']['pipeline_cost_usd']:.4f}")
    print(f"  Judge    : ${report['totals']['judge_cost_usd']:.4f}")
    print(f"  Total    : ${total:.4f}")


# ── main ──────────────────────────────────────────────────────────────────────

def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(message)s",
    )

    output_dir = Path(args.output_dir)
    (output_dir / "raw").mkdir(parents=True, exist_ok=True)
    (output_dir / "metrics").mkdir(parents=True, exist_ok=True)

    # ── judge-only mode ───────────────────────────────────────────────────────
    if args.judge_only:
        raw_dir = output_dir / "raw"
        jsonl_files = sorted(raw_dir.glob("*.jsonl"))
        if not jsonl_files:
            print(f"No .jsonl files found in {raw_dir}")
            sys.exit(0)
        print(f"Judge-only mode: {len(jsonl_files)} files to score")
        cost_rows = []
        for jf in jsonl_files:
            # infer pipeline/model from filename: {pipeline}_{model_safe}_{scale}.jsonl
            stem = jf.stem
            metrics_out = output_dir / "metrics" / f"{stem}_scored.csv"
            judge_cost = run_judge(jf, args.gold, metrics_out, args.judge_model, "gpt-4o-mini")
            cost_rows.append({"file": str(jf), "judge_cost_usd": judge_cost})
        aggregate_all(output_dir / "metrics", output_dir)
        write_cost_report(cost_rows, output_dir)
        return

    # ── resolve sweep dimensions ──────────────────────────────────────────────
    pipelines = _resolve(args.pipeline, ALL_PIPELINES)
    scales = [int(s) for s in _resolve(args.scale, ALL_SCALES)]
    models = _resolve(args.model, ALL_MODELS)

    tiers = [t.strip() for t in args.tier.split(",")] if args.tier else None
    query_ids = [q.strip() for q in args.query_ids.split(",")] if args.query_ids else None
    resume = not args.no_resume

    all_queries = load_queries(args.queries)
    selected_queries = select_queries(all_queries, query_ids=query_ids, tiers=tiers, limit=args.limit)
    gold_answers = load_gold_answers(args.gold if Path(args.gold).exists() else None)

    judge_model_for_dry = args.judge_model or get_recommended_judge(models[0])

    # ── dry run ───────────────────────────────────────────────────────────────
    if args.dry_run:
        dry_run(pipelines, scales, models, len(selected_queries), judge_model_for_dry)
        return

    # ── benchmark loop ────────────────────────────────────────────────────────
    combos = [(p, m, s) for m in models for s in scales for p in pipelines]
    print(f"Starting benchmark: {len(combos)} run(s), {len(selected_queries)} queries each")
    if not args.no_judge:
        print(f"Judging enabled (judge model auto-selected per evaluated model)")
    print()

    cost_rows = []

    for pipeline_name, model, scale in combos:
        corpus_path = str(Path(args.corpus_dir) / f"corpus_{scale}.jsonl")
        raw_path = _raw_path(args.output_dir, pipeline_name, model, scale)
        metrics_path = _metrics_path(args.output_dir, pipeline_name, model, scale)

        # resume logging
        completed_ids: set[str] = set()
        if resume and raw_path.exists():
            completed_ids = load_completed(str(raw_path))
            selected_ids = {q["id"] for q in selected_queries}
            done_in_selection = completed_ids & selected_ids
            remaining = len(selected_queries) - len(done_in_selection)
            if done_in_selection:
                print(
                    f"[{pipeline_name}/{_model_safe(model)}/scale={scale}] "
                    f"Resuming: {len(done_in_selection)}/{len(selected_queries)} already done, "
                    f"running {remaining} remaining."
                )
            if remaining <= 0:
                print(f"  All queries complete — skipping pipeline run, re-judging if needed.")
                if not args.no_judge:
                    judge_cost = run_judge(raw_path, args.gold, metrics_path, args.judge_model, model)
                    cost_rows.append({
                        "pipeline": pipeline_name, "model": model, "scale": scale,
                        "pipeline_cost_usd": 0.0, "judge_cost_usd": judge_cost,
                    })
                continue

        print(f"Running: {pipeline_name} | model={model} | scale={scale} | corpus={corpus_path}")

        try:
            pipeline = build_pipeline(pipeline_name, **_pipeline_kwargs(pipeline_name, model, args))
        except Exception as exc:
            print(f"  ERROR building pipeline: {exc}")
            continue

        records = run_pipeline(
            pipeline,
            corpus_path=corpus_path,
            queries=selected_queries,
            gold_answers=gold_answers,
            output_path=str(raw_path),
            resume=resume,
            show_progress=True,
            query_timeout=args.query_timeout or None,
        )

        # per-run cost tally
        pipeline_cost = sum(r.get("cost_usd") or 0.0 for r in records)
        ok_count = sum(1 for r in records if r.get("status") == "ok")
        err_count = sum(1 for r in records if r.get("status") == "error")
        print(
            f"  done: {ok_count} ok, {err_count} errors, "
            f"pipeline cost=${pipeline_cost:.4f} → {raw_path}"
        )

        # judge
        judge_cost = 0.0
        if not args.no_judge:
            judge_cost = run_judge(raw_path, args.gold, metrics_path, args.judge_model, model)

        cost_rows.append({
            "pipeline": pipeline_name,
            "model": model,
            "scale": scale,
            "pipeline_cost_usd": round(pipeline_cost, 6),
            "judge_cost_usd": round(judge_cost, 6),
        })

    # ── final aggregation ─────────────────────────────────────────────────────
    aggregate_all(output_dir / "metrics", output_dir)
    write_cost_report(cost_rows, output_dir)


if __name__ == "__main__":
    main()
