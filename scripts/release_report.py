"""Rebuild the frozen release's descriptive statistics without API calls."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def validate_judge_row(raw, score):
    if "judge call failed" in score.get("judge_reasoning", "").lower():
        raise ValueError("judge failures cannot be published as answer scores")
    local_guard = (raw.get("answer") == "EXCEEDS_CONTEXT"
                   and float(score["judge_score"]) == 0
                   and score.get("judge_reasoning", "").startswith("degenerate answer:"))
    if not local_guard and score.get("judge_model") not in {"gpt-4o", "", "nan"}:
        raise ValueError("unexpected judge: release must use one fixed judge")


def summarize(records):
    if not records or len({r["query_id"] for r in records}) != len(records):
        raise ValueError("empty or duplicate query records")
    infeasible = sum(r.get("answer") == "EXCEEDS_CONTEXT" for r in records)
    abstentions = sum(r.get("answer", "").strip() in
                      {"NO_FINAL_ANSWER", "UNKNOWN", "", "FINAL(UNKNOWN)"}
                      and r.get("status") == "ok" for r in records)
    return {
        "n": len(records),
        "mean_judge_score": None if infeasible == len(records) else
            float(np.mean([float(r["judge_score"]) for r in records])),
        "errors": sum(r.get("status") != "ok" for r in records),
        "infeasible": infeasible,
        "explicit_abstentions": abstentions,
        "mean_latency_seconds": float(np.mean([float(r.get("latency_ms") or 0) / 1000 for r in records])),
        "mean_cost_usd": float(np.mean([float(r.get("cost_usd") or 0) for r in records])),
        "total_cost_usd": sum(float(r.get("cost_usd") or 0) for r in records),
        "judge_cost_usd": sum(float(r.get("judge_cost_usd") or 0) for r in records),
        "index_cost_usd": max(float((r.get("trace") or {}).get("index_embedding_cost_usd") or 0) for r in records),
    }


def paired_interval(left, right):
    a = {r["query_id"]: float(r["judge_score"]) for r in left}
    b = {r["query_id"]: float(r["judge_score"]) for r in right}
    if not a or set(a) != set(b) or len(a) != len(left) or len(b) != len(right):
        raise ValueError("paired comparisons require the same unique query IDs")
    delta = np.array([a[q] - b[q] for q in sorted(a)])
    samples = np.random.default_rng(20260913).choice(delta, (10000, len(delta))).mean(axis=1)
    return {"difference": float(delta.mean()),
            "ci95": np.quantile(samples, [0.025, 0.975]).tolist()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", default="results/release_v1")
    parser.add_argument("--queries", default="data/release/v1/queries.json")
    args = parser.parse_args()
    root = Path(args.results)
    expected = {q["id"] for q in json.loads(Path(args.queries).read_text())}
    all_records = {}
    for pipeline in ("naive_llm", "vector_rag", "pageindex", "rlm"):
        stem = f"{pipeline}_gpt-4o-mini_150"
        raw = [json.loads(line) for line in (root / "raw" / f"{stem}.jsonl").read_text().splitlines()]
        if len(raw) != len(expected) or {r["query_id"] for r in raw} != expected:
            raise ValueError(f"{pipeline}: incomplete or duplicate release rows")
        with (root / "metrics" / f"{stem}_scored.csv").open() as handle:
            scored = list(csv.DictReader(handle))
        if len(scored) != len(expected) or {r["query_id"] for r in scored} != expected:
            raise ValueError(f"{pipeline}: incomplete or duplicate scores")
        scores = {r["query_id"]: r for r in scored}
        for r in raw:
            score = scores[r["query_id"]]
            validate_judge_row(r, score)
            r.update(judge_score=float(score["judge_score"]),
                     judge_cost_usd=float(score["judge_cost_usd"]),
                     judge_reasoning=score["judge_reasoning"])
        all_records[pipeline] = raw
    report = {"pipelines": {name: summarize(rows) for name, rows in all_records.items()},
              "paired_pageindex_minus_vector": paired_interval(all_records["pageindex"], all_records["vector_rag"]),
              "paired_rlm_minus_vector": paired_interval(all_records["rlm"], all_records["vector_rag"]),
              "uncertainty_note": "Exploratory paired query bootstrap, 10,000 draws, fixed seed. Questions sharing companies are dependent; intervals are not population-level significance claims.",
              "by_tier": {name: {tier: summarize([r for r in rows if r["tier_name"] == tier])
                                 for tier in sorted({r["tier_name"] for r in rows})}
                          for name, rows in all_records.items()}}
    (root / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    (root / "review_records.json").write_text(json.dumps(all_records, indent=2) + "\n")
    print(json.dumps(report["pipelines"], indent=2))


if __name__ == "__main__":
    main()
