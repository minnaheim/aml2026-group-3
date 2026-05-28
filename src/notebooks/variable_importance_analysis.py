"""
Variable importance analysis for TFT interpretation outputs.

Reads out/holdout/interpretation/{TARGET}/var_selection_{TARGET}_h{horizon}_{emb_tag}.csv,
aggregates importance across targets per group, and identifies candidates for removal.

Usage:
    python src/notebooks/variable_importance_analysis.py
    python src/notebooks/variable_importance_analysis.py --horizon 12 --emb-tag none --threshold 0.005
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

TARGETS   = ["CPI", "GDP", "UNRATE"]
VAR_TYPES = ["encoder_variables", "decoder_variables", "static_variables"]

# Only encoder variables are actionable (decoder = known calendar/FOMC; static = metadata)
ACTIONABLE = {"encoder_variables"}


def load_all(interp_dir: Path, horizon: int, emb_tag: str) -> dict[str, pd.DataFrame]:
    """Return {group: DataFrame(variable, CPI, GDP, UNRATE, mean_importance, min_importance)}."""
    frames: dict[str, list[pd.DataFrame]] = {vt: [] for vt in VAR_TYPES}

    for target in TARGETS:
        fname = f"var_selection_{target}_h{horizon}_{emb_tag}.csv"
        path  = interp_dir / target / fname
        if not path.exists():
            print(f"  [WARN] missing: {path}")
            continue
        df = pd.read_csv(path)
        for group, grp_df in df.groupby("group"):
            if group not in frames:
                continue
            frames[group].append(
                grp_df[["variable", "importance"]]
                .set_index("variable")
                .rename(columns={"importance": target})
            )

    merged = {}
    for vt, dfs in frames.items():
        if not dfs:
            continue
        combined = pd.concat(dfs, axis=1).fillna(0.0)
        present  = [t for t in TARGETS if t in combined.columns]
        combined["mean_importance"] = combined[present].mean(axis=1)
        combined["min_importance"]  = combined[present].min(axis=1)
        combined = combined.sort_values("mean_importance", ascending=False).reset_index()
        merged[vt] = combined
    return merged


def flag_candidates(df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    return df[df["mean_importance"] < threshold].sort_values("mean_importance")


def plot_importance(df: pd.DataFrame, vt: str, out_path: Path, threshold: float) -> None:
    top = df.head(30)
    fig, ax = plt.subplots(figsize=(10, max(4, len(top) * 0.35)))
    colors = ["#d62728" if v < threshold else "#1f77b4" for v in top["mean_importance"]]
    ax.barh(top["variable"][::-1], top["mean_importance"][::-1], color=colors[::-1])
    ax.axvline(threshold, color="red", linestyle="--", linewidth=1, label=f"threshold={threshold}")
    ax.set_xlabel("Mean importance across targets")
    ax.set_title(f"{vt.replace('_', ' ').title()} — mean importance (red = below threshold)")
    ax.legend()
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Plot saved → {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--horizon",   type=int,   default=12)
    parser.add_argument("--emb-tag",   default="none")
    parser.add_argument("--threshold", type=float, default=0.005,
                        help="Mean importance below this → removal candidate (default: 0.005)")
    args = parser.parse_args()

    root       = Path(__file__).parent.parent.parent
    interp_dir = root / "out" / "holdout" / "interpretation"
    out_dir    = root / "out" / "holdout" / "interpretation" / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nHorizon  : h{args.horizon}")
    print(f"Emb tag  : {args.emb_tag}")
    print(f"Threshold: {args.threshold}\n")

    all_data = load_all(interp_dir, args.horizon, args.emb_tag)

    removal_candidates: list[str] = []

    for vt in VAR_TYPES:
        if vt not in all_data:
            continue
        df = all_data[vt]
        present = [t for t in TARGETS if t in df.columns]
        display_cols = ["variable"] + present + ["mean_importance", "min_importance"]

        print(f"{'═' * 60}")
        print(f"  {vt.replace('_', ' ').upper()}")
        print(f"{'═' * 60}")
        print(df[display_cols].to_string(index=False, float_format=lambda x: f"{x:.5f}"))

        candidates = flag_candidates(df, args.threshold)
        if candidates.empty:
            print(f"\n  No variables below threshold={args.threshold}")
        else:
            print(f"\n  Removal candidates (mean_importance < {args.threshold}):")
            print(candidates[display_cols].to_string(index=False, float_format=lambda x: f"{x:.5f}"))
            if vt in ACTIONABLE:
                removal_candidates.extend(candidates["variable"].tolist())

        csv_path = out_dir / f"importance_{vt}_h{args.horizon}_{args.emb_tag}.csv"
        df[display_cols].to_csv(csv_path, index=False)
        print(f"\n  CSV saved → {csv_path}")

        if vt == "encoder_variables":
            plot_importance(df, vt, out_dir / f"importance_{vt}_h{args.horizon}_{args.emb_tag}.png",
                            threshold=args.threshold)
        print()

    # ── summary ──────────────────────────────────────────────────────────────
    print(f"{'═' * 60}")
    print("  ENCODER REMOVAL CANDIDATES (mean importance < threshold)")
    print(f"{'═' * 60}")
    if removal_candidates:
        for v in removal_candidates:
            print(f"    {v}")
        print(f"\n  → {len(removal_candidates)} variable(s) to consider removing from")
        print(f"     time_varying_unknown_reals in TFTRunner.create_tft_dataset()")
    else:
        print("  None — all encoder variables exceed threshold.")

    summary_path = out_dir / f"removal_candidates_h{args.horizon}_{args.emb_tag}.txt"
    summary_path.write_text("\n".join(removal_candidates))
    print(f"\n  Summary saved → {summary_path}")


if __name__ == "__main__":
    main()
