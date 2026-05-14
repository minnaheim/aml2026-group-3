#!/usr/bin/env python3
"""
Plot ablation study results from experiments.csv.

Usage:
    python plot_experiments.py
    python plot_experiments.py --csv out/holdout/v20260514/experiments.csv
    python plot_experiments.py --csv out/holdout/v20260514/experiments.csv --metric MAE --targets CPI
    python plot_experiments.py --exclude-runs bad_run1 bad_run2
"""
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from pathlib import Path

AGGREGATION_LABELS = {
    "mean":              "Mean",
    "decay":             "Exp. Decay",
    "attention":         "Attention",
    "context_attention": "Context Attn",
}

TFT_COLORS = [
    "#2980B9", "#1ABC9C", "#8E44AD", "#D35400",
    "#27AE60", "#F39C12", "#2C3E50", "#16A085",
]


def load_experiments(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df["embedding"]   = df["embedding"].fillna("none")
    df["aggregation"] = df["aggregation"].fillna("mean")
    return df


def make_tft_label(row) -> str:
    if row["embedding"] == "none":
        return "TFT (macro only)"
    agg = AGGREGATION_LABELS.get(row["aggregation"], row["aggregation"])
    return f"TFT | {row['embedding']} | {agg}"


def plot_comparison(df, metric, targets, out_path, exclude_runs=None):
    """
    One subplot per target.
    Baselines (AR, ARIMA) shown once at top, then each TFT variant below.
    """
    exclude_runs = exclude_runs or []
    n_targets = len(targets)
    fig, axes = plt.subplots(1, n_targets, figsize=(7 * n_targets, 7), squeeze=False)
    fig.suptitle(f"Model Comparison — {metric}", fontsize=14, fontweight="bold", y=1.02)

    for col, target in enumerate(targets):
        ax = axes[0][col]
        sub = df[(df["target"] == target) & (~df["run_name"].isin(exclude_runs))].copy()

        rows = []

        # AR and ARIMA — deduplicated (same across runs)
        for model, color in [("AR", "#C0392B"), ("ARIMA", "#E67E22")]:
            model_rows = sub[sub["model"] == model]
            if len(model_rows) > 0:
                rows.append({
                    "label": model,
                    "value": float(model_rows[metric].iloc[0]),
                    "color": color,
                    "is_baseline": True,
                })

        # TFT variants — one per unique run, sorted best to worst (ascending for hbar)
        tft_rows = sub[sub["model"] == "TFT"].copy()
        tft_rows["label"] = tft_rows.apply(make_tft_label, axis=1)
        # drop duplicates (same label = same config run twice)
        tft_rows = tft_rows.drop_duplicates("label")
        tft_rows = tft_rows.sort_values(metric, ascending=True)  # best at bottom for hbar

        for i, (_, row) in enumerate(tft_rows.iterrows()):
            rows.append({
                "label": row["label"],
                "value": float(row[metric]),
                "color": TFT_COLORS[i % len(TFT_COLORS)],
                "is_baseline": False,
            })

        if not rows:
            ax.set_visible(False)
            continue

        plot_df = pd.DataFrame(rows)
        labels  = plot_df["label"].tolist()
        values  = plot_df["value"].tolist()
        colors  = plot_df["color"].tolist()

        bars = ax.barh(labels, values, color=colors, edgecolor="white",
                       linewidth=0.6, height=0.6)

        max_val = max(values) if values else 1
        for bar, val in zip(bars, values):
            ax.text(
                bar.get_width() + max_val * 0.01,
                bar.get_y() + bar.get_height() / 2,
                f"{val:.4f}", va="center", ha="left", fontsize=8,
            )

        # separator line between baselines and TFT variants
        n_baselines = sum(1 for r in rows if r["is_baseline"])
        if 0 < n_baselines < len(rows):
            ax.axhline(y=n_baselines - 0.5, color="gray", linestyle="--",
                       linewidth=0.8, alpha=0.6)

        ax.set_title(target, fontsize=12, fontweight="bold")
        ax.set_xlabel(metric)
        ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.4f"))
        ax.grid(axis="x", alpha=0.3, linestyle="--")
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_xlim(right=max_val * 1.18)

    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"Saved → {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Plot ablation study results")
    parser.add_argument("--csv",     default="out/holdout/experiments.csv")
    parser.add_argument("--metric",  default="MAE", choices=["MAE", "RMSE"])
    parser.add_argument("--targets", nargs="+", default=["CPI", "UNRATE", "GDP"])
    parser.add_argument("--outdir",  default="out/holdout/plots")
    parser.add_argument("--exclude-runs", nargs="+", default=[],
                        help="Run names to exclude (e.g. problematic runs)")
    args = parser.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        print(f"ERROR: {csv_path} not found.")
        return

    out_dir = Path(args.outdir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_experiments(csv_path)
    print(f"Loaded {len(df)} rows from {csv_path}")
    print(df[["run_name", "model", "target", "embedding", "aggregation", args.metric]].to_string(index=False))

    plot_comparison(
        df, args.metric, args.targets,
        out_dir / f"comparison_{args.metric.lower()}.png",
        exclude_runs=args.exclude_runs,
    )

    # also plot RMSE if MAE was requested (and vice versa)
    other_metric = "RMSE" if args.metric == "MAE" else "MAE"
    plot_comparison(
        df, other_metric, args.targets,
        out_dir / f"comparison_{other_metric.lower()}.png",
        exclude_runs=args.exclude_runs,
    )

    print(f"\nPlots saved to {out_dir}")


if __name__ == "__main__":
    main()