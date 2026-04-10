#!/usr/bin/env python3
"""Generate publication-quality charts from benchmark results.

Usage:
    python scripts/generate_plots.py
    python scripts/generate_plots.py --metrics-dir results/metrics --output-dir plots
    python scripts/generate_plots.py --charts accuracy_vs_scale,cost_per_query
    python scripts/generate_plots.py --style seaborn-v0_8-whitegrid
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

# ── project root ──────────────────────────────────────────────────────────────
_ROOT = Path(__file__).parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# ── pipeline display names and color palette ──────────────────────────────────
PIPELINE_ORDER = ["naive_llm", "vector_rag", "pageindex", "rlm"]
PIPELINE_LABELS = {
    "naive_llm": "Naive LLM",
    "vector_rag": "Vector RAG",
    "pageindex": "PageIndex",
    "rlm": "RLM",
}
PIPELINE_COLORS = {
    "naive_llm": "#888888",
    "vector_rag": "#2196F3",
    "pageindex": "#4CAF50",
    "rlm": "#FF9800",
}

SCALE_ORDER = [10, 25, 50, 100, 150]

ALL_CHARTS = [
    "accuracy_vs_scale",
    "accuracy_by_tier_heatmap",
    "cost_per_query_bar",
    "latency_distribution",
    "accuracy_cost_scatter",
]


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate benchmark visualizations.")
    p.add_argument("--metrics-dir", default="results/metrics")
    p.add_argument("--output-dir", default="plots")
    p.add_argument(
        "--charts",
        default="all",
        help="Comma-separated chart names, or 'all'.",
    )
    p.add_argument("--style", default="seaborn-v0_8-whitegrid")
    p.add_argument("--dpi", type=int, default=300)
    p.add_argument("--no-svg", action="store_true", help="Skip SVG output.")
    return p.parse_args(argv)


# ── data loading ──────────────────────────────────────────────────────────────

def load_scored_data(metrics_dir: Path):
    import pandas as pd  # type: ignore[import-untyped]

    csv_files = sorted(metrics_dir.glob("*_scored.csv"))
    if not csv_files:
        # also accept judged CSVs
        csv_files = sorted(metrics_dir.glob("*_judged.csv"))
    if not csv_files:
        sys.exit(f"No scored CSV files found in {metrics_dir}. Run the benchmark first.")

    dfs = []
    for f in csv_files:
        try:
            dfs.append(pd.read_csv(f))
        except Exception as exc:
            warnings.warn(f"Skipping {f}: {exc}")
    if not dfs:
        sys.exit("Could not load any CSV files.")

    df = pd.concat(dfs, ignore_index=True)

    # normalise score column — accept either judge_score or llm_judge
    if "judge_score" not in df.columns and "llm_judge" in df.columns:
        df["judge_score"] = df["llm_judge"]
    if "judge_score" not in df.columns:
        sys.exit("No judge_score or llm_judge column found in scored CSVs.")

    # ensure numeric
    for col in ["judge_score", "cost_usd", "latency_ms", "scale"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


# ── chart helpers ─────────────────────────────────────────────────────────────

def _save(fig, path_stem: Path, dpi: int, no_svg: bool) -> None:
    path_stem.parent.mkdir(parents=True, exist_ok=True)
    png_path = path_stem.with_suffix(".png")
    fig.savefig(png_path, dpi=dpi, bbox_inches="tight")
    print(f"  Saved {png_path}")
    if not no_svg:
        svg_path = path_stem.with_suffix(".svg")
        fig.savefig(svg_path, bbox_inches="tight")
        print(f"  Saved {svg_path}")


def _pipeline_label(name: str) -> str:
    return PIPELINE_LABELS.get(name, name)


def _pipeline_color(name: str) -> str:
    return PIPELINE_COLORS.get(name, "#333333")


def _ordered_pipelines(df) -> list[str]:
    present = df["pipeline"].dropna().unique().tolist()
    ordered = [p for p in PIPELINE_ORDER if p in present]
    ordered += [p for p in present if p not in ordered]
    return ordered


# ── Chart 1: accuracy vs scale ────────────────────────────────────────────────

def chart_accuracy_vs_scale(df, output_dir: Path, dpi: int, no_svg: bool) -> None:
    import matplotlib.pyplot as plt  # type: ignore[import-untyped]

    print("Generating accuracy_vs_scale...")
    models = df["model"].dropna().unique().tolist()
    n_models = len(models)
    fig_w = 10 * n_models if n_models > 1 else 10
    fig, axes = plt.subplots(1, n_models, figsize=(fig_w, 6), squeeze=False)

    for ax, model in zip(axes[0], models):
        mdf = df[df["model"] == model]
        pipelines = _ordered_pipelines(mdf)
        for pipeline in pipelines:
            pdf = (
                mdf[mdf["pipeline"] == pipeline]
                .groupby("scale")["judge_score"]
                .mean()
                .reset_index()
                .sort_values("scale")
            )
            if pdf.empty:
                continue
            ax.plot(
                pdf["scale"],
                pdf["judge_score"],
                marker="o",
                linewidth=2,
                markersize=7,
                label=_pipeline_label(pipeline),
                color=_pipeline_color(pipeline),
            )

        ax.set_xlabel("Corpus size (# documents)", fontsize=13)
        ax.set_ylabel("Mean judge score", fontsize=13)
        ax.set_title(f"Accuracy vs. Scale — {model}", fontsize=14, fontweight="bold")
        ax.set_ylim(0, 1.05)
        ax.set_xticks(SCALE_ORDER)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.4)

    fig.tight_layout()
    _save(fig, output_dir / "accuracy_vs_scale", dpi, no_svg)
    plt.close(fig)


# ── Chart 2: accuracy by tier heatmap ────────────────────────────────────────

def chart_accuracy_by_tier_heatmap(df, output_dir: Path, dpi: int, no_svg: bool) -> None:
    import matplotlib.pyplot as plt  # type: ignore[import-untyped]
    import seaborn as sns  # type: ignore[import-untyped]

    print("Generating accuracy_by_tier_heatmap...")
    if "tier_name" not in df.columns or df["tier_name"].isna().all():
        print("  Skipping: no tier_name column.")
        return

    pipelines = _ordered_pipelines(df)
    pivot = (
        df.groupby(["pipeline", "tier_name"])["judge_score"]
        .mean()
        .unstack()
        .reindex(pipelines)
    )
    pivot.index = [_pipeline_label(p) for p in pivot.index]

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.heatmap(
        pivot,
        ax=ax,
        annot=True,
        fmt=".2f",
        cmap="RdYlGn",
        vmin=0,
        vmax=1,
        linewidths=0.5,
        annot_kws={"size": 12},
        cbar_kws={"label": "Mean judge score"},
    )
    ax.set_title("Accuracy by Pipeline × Tier", fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("Tier", fontsize=13)
    ax.set_ylabel("Pipeline", fontsize=13)
    ax.tick_params(axis="both", labelsize=11)
    fig.tight_layout()
    _save(fig, output_dir / "accuracy_by_tier_heatmap", dpi, no_svg)
    plt.close(fig)


# ── Chart 3: cost per query bar ───────────────────────────────────────────────

def chart_cost_per_query_bar(df, output_dir: Path, dpi: int, no_svg: bool) -> None:
    import matplotlib.pyplot as plt  # type: ignore[import-untyped]
    import numpy as np  # type: ignore[import-untyped]

    print("Generating cost_per_query_bar...")
    if "cost_usd" not in df.columns or df["cost_usd"].isna().all():
        print("  Skipping: no cost_usd data.")
        return

    cost = (
        df.groupby(["pipeline", "scale"])["cost_usd"]
        .mean()
        .reset_index()
    )
    pipelines = _ordered_pipelines(df)
    scales = sorted(cost["scale"].dropna().unique().tolist())

    x = np.arange(len(pipelines))
    bar_w = 0.8 / max(len(scales), 1)
    fig, ax = plt.subplots(figsize=(11, 6))

    for i, scale in enumerate(scales):
        sdf = cost[cost["scale"] == scale].set_index("pipeline")
        vals = [sdf.loc[p, "cost_usd"] if p in sdf.index else 0.0 for p in pipelines]
        offset = (i - len(scales) / 2 + 0.5) * bar_w
        bars = ax.bar(
            x + offset,
            vals,
            width=bar_w * 0.9,
            label=f"Scale {int(scale)}",
        )

    ax.set_xticks(x)
    ax.set_xticklabels([_pipeline_label(p) for p in pipelines], fontsize=12)
    ax.set_xlabel("Pipeline", fontsize=13)
    ax.set_ylabel("Avg cost per query (USD)", fontsize=13)
    ax.set_title("Cost per Query by Pipeline and Scale", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10, ncol=min(len(scales), 5))
    ax.grid(axis="y", alpha=0.4)
    fig.tight_layout()
    _save(fig, output_dir / "cost_per_query_bar", dpi, no_svg)
    plt.close(fig)


# ── Chart 4: latency distribution ────────────────────────────────────────────

def chart_latency_distribution(df, output_dir: Path, dpi: int, no_svg: bool) -> None:
    import matplotlib.pyplot as plt  # type: ignore[import-untyped]

    print("Generating latency_distribution...")
    if "latency_ms" not in df.columns or df["latency_ms"].isna().all():
        print("  Skipping: no latency_ms data.")
        return

    pipelines = _ordered_pipelines(df)
    data = [
        df.loc[df["pipeline"] == p, "latency_ms"].dropna().tolist()
        for p in pipelines
    ]
    data = [d for d in data if d]  # drop empty
    labels = [_pipeline_label(p) for p, d in zip(pipelines, data) if d]
    colors = [_pipeline_color(p) for p, d in zip(pipelines, data) if d]
    pipelines = [p for p, d in zip(pipelines, data) if d]

    fig, ax = plt.subplots(figsize=(10, 6))
    bp = ax.boxplot(
        data,
        labels=labels,
        patch_artist=True,
        medianprops={"color": "black", "linewidth": 2},
    )
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    # log scale if max/min ratio is large
    lats_flat = [v for d in data for v in d]
    if lats_flat and max(lats_flat) / max(min(lats_flat), 1) > 100:
        ax.set_yscale("log")
        ax.set_ylabel("Latency (ms, log scale)", fontsize=13)
    else:
        ax.set_ylabel("Latency (ms)", fontsize=13)

    ax.set_xlabel("Pipeline", fontsize=13)
    ax.set_title("Query Latency Distribution by Pipeline", fontsize=14, fontweight="bold")
    ax.grid(axis="y", alpha=0.4)
    fig.tight_layout()
    _save(fig, output_dir / "latency_distribution", dpi, no_svg)
    plt.close(fig)


# ── Chart 5: accuracy vs cost scatter ────────────────────────────────────────

def chart_accuracy_cost_scatter(df, output_dir: Path, dpi: int, no_svg: bool) -> None:
    import matplotlib.pyplot as plt  # type: ignore[import-untyped]
    import numpy as np  # type: ignore[import-untyped]

    print("Generating accuracy_cost_scatter...")
    if "cost_usd" not in df.columns or df["cost_usd"].isna().all():
        print("  Skipping: no cost_usd data.")
        return

    plot_df = df[["pipeline", "judge_score", "cost_usd"]].dropna()
    if plot_df.empty:
        print("  Skipping: no complete rows.")
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    pipelines = _ordered_pipelines(df)
    for pipeline in pipelines:
        pdf = plot_df[plot_df["pipeline"] == pipeline]
        if pdf.empty:
            continue
        ax.scatter(
            pdf["cost_usd"],
            pdf["judge_score"],
            label=_pipeline_label(pipeline),
            color=_pipeline_color(pipeline),
            alpha=0.5,
            s=30,
        )

    # Pareto frontier (maximise score, minimise cost)
    agg = plot_df.groupby("pipeline").agg(
        cost=("cost_usd", "mean"),
        score=("judge_score", "mean"),
    ).reset_index().sort_values("cost")

    pareto_pts: list[tuple[float, float]] = []
    best_score = -1.0
    for _, row in agg.sort_values("cost").iterrows():
        if row["score"] > best_score:
            best_score = row["score"]
            pareto_pts.append((row["cost"], row["score"]))
    if len(pareto_pts) > 1:
        px, py = zip(*pareto_pts)
        ax.plot(px, py, "k--", linewidth=1.5, label="Pareto front", alpha=0.7)

    ax.set_xlabel("Cost per query (USD)", fontsize=13)
    ax.set_ylabel("Judge score", fontsize=13)
    ax.set_title("Accuracy vs. Cost Trade-off", fontsize=14, fontweight="bold")
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.4)
    fig.tight_layout()
    _save(fig, output_dir / "accuracy_cost_scatter", dpi, no_svg)
    plt.close(fig)


# ── Summary tables ────────────────────────────────────────────────────────────

def write_summary_tables(df, output_dir: Path) -> None:
    import pandas as pd  # type: ignore[import-untyped]

    out = output_dir / "tables"
    out.mkdir(parents=True, exist_ok=True)

    # accuracy by tier
    if "tier_name" in df.columns:
        acc = (
            df.groupby(["pipeline", "model", "scale", "tier_name"])["judge_score"]
            .agg(mean_score="mean", std_score="std", n_queries="count")
            .reset_index()
            .round(4)
        )
        p = out / "accuracy_by_tier.csv"
        acc.to_csv(p, index=False)
        print(f"  Saved {p}")

    # cost comparison
    cost_cols = [c for c in ["cost_usd", "tokens_in", "tokens_out"] if c in df.columns]
    if cost_cols:
        agg_dict: dict = {}
        if "cost_usd" in df.columns:
            agg_dict["avg_cost_usd"] = ("cost_usd", "mean")
            agg_dict["total_cost_usd"] = ("cost_usd", "sum")
        cost_df = (
            df.groupby(["pipeline", "model", "scale"])
            .agg(**agg_dict)
            .reset_index()
            .round(6)
        )
        p = out / "cost_comparison.csv"
        cost_df.to_csv(p, index=False)
        print(f"  Saved {p}")

    # latency summary
    if "latency_ms" in df.columns:
        lat = (
            df.groupby(["pipeline", "model", "scale"])["latency_ms"]
            .agg(
                p50_ms=lambda x: x.quantile(0.50),
                p95_ms=lambda x: x.quantile(0.95),
                p99_ms=lambda x: x.quantile(0.99),
                mean_ms="mean",
            )
            .reset_index()
            .round(1)
        )
        p = out / "latency_summary.csv"
        lat.to_csv(p, index=False)
        print(f"  Saved {p}")


# ── main ──────────────────────────────────────────────────────────────────────

def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        try:
            plt.style.use(args.style)
        except OSError:
            plt.style.use("seaborn-v0_8-whitegrid")
        plt.rcParams.update({
            "font.size": 12,
            "axes.titlesize": 14,
            "axes.labelsize": 13,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "legend.fontsize": 11,
        })
    except ImportError:
        sys.exit("matplotlib is required: pip install matplotlib")

    metrics_dir = Path(args.metrics_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading scored results from {metrics_dir} ...")
    df = load_scored_data(metrics_dir)
    print(f"  {len(df)} rows across {df['pipeline'].nunique()} pipeline(s), "
          f"{df['scale'].nunique()} scale(s)")

    charts = (
        ALL_CHARTS
        if args.charts.strip().lower() == "all"
        else [c.strip() for c in args.charts.split(",")]
    )

    print(f"\nGenerating charts → {output_dir}")
    if "accuracy_vs_scale" in charts:
        chart_accuracy_vs_scale(df, output_dir, args.dpi, args.no_svg)
    if "accuracy_by_tier_heatmap" in charts:
        chart_accuracy_by_tier_heatmap(df, output_dir, args.dpi, args.no_svg)
    if "cost_per_query_bar" in charts:
        chart_cost_per_query_bar(df, output_dir, args.dpi, args.no_svg)
    if "latency_distribution" in charts:
        chart_latency_distribution(df, output_dir, args.dpi, args.no_svg)
    if "accuracy_cost_scatter" in charts:
        chart_accuracy_cost_scatter(df, output_dir, args.dpi, args.no_svg)

    print("\nGenerating summary tables ...")
    write_summary_tables(df, output_dir)
    print("\nDone.")


if __name__ == "__main__":
    main()
