#!/usr/bin/env python3
"""
Holdout validation pipeline: AR(1) benchmark vs TFT.

Usage:
    python main.py
    python main.py --targets CPI GDP --device mps --wandb
    python main.py --targets UNRATE --device cuda
"""
import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless-safe; no display required
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import wandb
import os

sys.path.insert(0, str(Path(__file__).parent))
from data_frame_builder  import DataFrameBuilder
from benchmark_runner    import ARRunner, ARIMARunner
from tft_runner          import TFTRunner
from embedding_manager   import EmbeddingManager

MAIN_TARGETS = ["CPI", "UNRATE", "GDP" ]
ALL_TARGETS = ["CPI", "PAYEMS", "INDPRO", "UNRATE", "GDP"]

# --- weights and biases logging ------------------------

_RENKU_WANDB_SECRET = Path("/secrets/wandb-api-key.txt")

def _setup_wandb() -> None:
    """Log in to W&B.
     with the session secret, in renku, else you input it yourself
    """
    if not os.environ.get("WANDB_API_KEY"):
        if _RENKU_WANDB_SECRET.exists():
            os.environ["WANDB_API_KEY"] = _RENKU_WANDB_SECRET.read_text().strip()
            print(f"W&B: loaded API key from {_RENKU_WANDB_SECRET}")
        else:
            key = input("Enter your W&B API key (or set WANDB_API_KEY / mount secret): ").strip()
            os.environ["WANDB_API_KEY"] = key
    wandb.login()


# ── metrics ──────────────────────────────────────────────────────────────────

def compute_metrics(df: pd.DataFrame) -> dict:
    errors = df["actual"].values - df["predicted"].values
    mae  = float(np.mean(np.abs(errors)))
    rmse = float(np.sqrt(np.mean(errors ** 2)))
    mask = df["actual"].values != 0
    mape = float(np.mean(np.abs(errors[mask] / df["actual"].values[mask])) * 100)
    return {"MAE": round(mae, 4), "RMSE": round(rmse, 4), "MAPE": round(mape, 4)}


# ── save ─────────────────────────────────────────────────────────────────────

def save_results(results: dict, out_dir: Path) -> pd.DataFrame:
    """Save per-fold CSVs + per-fold metrics + fold-averaged metrics."""
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for model, fold_res in results.items():
        for fold, target_res in fold_res.items():
            for target, df in target_res.items():
                df.to_csv(out_dir / f"{model.lower()}_{target.lower()}_fold{fold}_predictions.csv", index=False)
                m = compute_metrics(df)
                rows.append({"model": model, "fold": fold, "target": target, **m})

    metrics_df = pd.DataFrame(rows)[["model", "fold", "target", "MAE", "RMSE", "MAPE"]]
    metrics_df = metrics_df.sort_values(by=["target", "model", "fold"]).reset_index(drop=True)
    metrics_df.to_csv(out_dir / "metrics_per_fold.csv", index=False)

    avg_df = (
        metrics_df.groupby(["model", "target"])[["MAE", "RMSE", "MAPE"]]
        .mean().round(4).reset_index()
        .sort_values(by=["target", "model"]).reset_index(drop=True)
    )
    avg_df.to_csv(out_dir / "metrics.csv", index=False)

    print("\n=== Evaluation Metrics (per fold) ===")
    print(metrics_df.to_string(index=False))
    print("\n=== Evaluation Metrics (averaged across folds) ===")
    print(avg_df.to_string(index=False))
    return avg_df


# ── plot ──────────────────────────────────────────────────────────────────────

def plot_results(results: dict, out_dir: Path):
    # collect all targets across all folds
    targets = sorted({t for fold_res in results.values() for tr in fold_res.values() for t in tr})
    _, axes = plt.subplots(len(targets), 1, figsize=(14, 4 * len(targets)), squeeze=False)

    colors     = {"AR": "crimson", "ARIMA": "darkorange", "TFT": "steelblue"}
    linestyles = {"AR": "--",      "ARIMA": "-.",          "TFT": ":"}

    for i, target in enumerate(targets):
        ax = axes[i][0]
        actual_drawn = False

        for model, fold_res in results.items():
            # concatenate predictions across folds (non-overlapping windows → one continuous series)
            dfs = [fold_res[f][target] for f in sorted(fold_res) if target in fold_res[f]]
            if not dfs:
                continue
            df = pd.concat(dfs).sort_values("date").dropna(subset=["actual", "predicted"])

            if not actual_drawn:
                ax.plot(df["date"], df["actual"],
                        color="black", linewidth=1.5, label="Actual")
                actual_drawn = True

            ax.plot(df["date"], df["predicted"],
                    color=colors.get(model, "orange"),
                    linewidth=1.5,
                    linestyle=linestyles.get(model, "-"),
                    label=f"{model} Predicted")

        ax.set_title(f"{target}: Predicted vs Actual")
        ax.set_xlabel("Date")
        ax.legend()
        ax.grid(alpha=0.3)

    plt.tight_layout()
    path = out_dir / "predictions_vs_actuals.png"
    plt.savefig(path, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"Plot saved → {path}")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    # parse arguments passed by user
    parser = argparse.ArgumentParser(
        description="Holdout validation pipeline: AR(1) benchmark vs TFT"
    )
    parser.add_argument(
        "--targets", nargs="+", default=MAIN_TARGETS,
        choices=MAIN_TARGETS, metavar="TARGET",
        help=f"Targets to forecast. One or more of: {MAIN_TARGETS}  (default: all)",
    )
    parser.add_argument(
        "--wandb", action="store_true", default=False,
        help="Enable Weights & Biases logging for TFT training",
    )
    parser.add_argument(
        "--device", default="cpu", choices=["cpu", "mps", "cuda"],
        help="Compute device for TFT (default: cpu)",
    )
    parser.add_argument(
        "--embedding", default=None,
        choices=["fomc-roberta", "finbert", "finbert-kafka", "fomc-roberta-kafka", "roberta", "llama3.1"],
        help="Speech embedding to include (default: none — macro-only mode)",
    )
    
    parser.add_argument(
        "--aggregation", default="mean", choices=["mean", "decay", "attention", "context_attention"],
        help="Speech aggregation strategy (default: mean)",
    )
    
    parser.add_argument(
        "--horizon", type=int, default=12, choices=[1, 6, 9, 12],
        help="Forecast horizon in months (default: 12)",
    )
    args = parser.parse_args()

    # make sure that there is no sneaky cuda being used 
    if args.device != "cuda":                                                                                                                                            
          os.environ["CUDA_VISIBLE_DEVICES"] = ""   

    root    = Path(__file__).parent.parent.parent
    out_dir = root / "out" / "holdout"

    if args.wandb:
        _setup_wandb()

    print(f"Targets   : {args.targets}")
    print(f"Device    : {args.device}")
    print(f"Embedding : {args.embedding or 'none'}")
    print(f"Horizon   : {args.horizon}")
    print(f"W&B       : {args.wandb}")
    print(f"Output    : {out_dir}\n")

    # ── 1. split the data acc. to data-frame-builder ─────────────────────────
    # dissents are loaded explicitly so those features appear in process_data()
    # regardless of whether embeddings are used
    dfb = DataFrameBuilder(str(root), aggregation=args.aggregation)
    dfb.load_fomc_dissent()
    df  = dfb.process_data()
    splits, holdout = dfb.generate_split(df)

    if args.embedding is not None: # allow for different models!
        # re-fit PCA per fold on training speeches only — no look-ahead leakage
        emb = EmbeddingManager(str(root), embedding=args.embedding).load()
        splits = dfb.add_leakage_free_embeddings(splits, emb)

    for s in splits:
        tr, te = s["train"], s["test"]
        print(
            f"Fold {s['fold']}: "
            f"train [{tr['date'].min().date()} – {tr['date'].max().date()}] ({len(tr)} rows) | "
            f"test  [{te['date'].min().date()} – {te['date'].max().date()}] ({len(te)} rows)"
        )
    print(
        f"Holdout : [{holdout['date'].min().date()} – {holdout['date'].max().date()}] "
        f"({len(holdout)} rows)\n"
    )
    # initiate runners
    ar_runner    = ARRunner(dfb)
    arima_runner = ARIMARunner(dfb)
    tft_runner   = TFTRunner(dfb)

    # results[model][fold_num][target] = predictions DataFrame
    results: dict[str, dict[int, dict[str, pd.DataFrame]]] = {
        "AR":    {s["fold"]: {} for s in splits},
        "ARIMA": {s["fold"]: {} for s in splits},
        "TFT":   {s["fold"]: {} for s in splits},
    }

    # ── 2. run all models for each fold × target ──────────────────────────────
    for fold_idx, s in enumerate(splits):
        fold_num = s["fold"]
        for target in args.targets:
            print(f"\n{'─' * 55}")
            print(f" Fold {fold_num} | Target: {target}")
            print(f"{'─' * 55}")

            print("\n[AR]")
            ar_order, ar_seasonal = ar_runner.run(splits, target=target, fold=fold_idx)
            ar_df = ar_runner.predict(splits, target=target, fold=fold_idx, step=args.horizon)
            results["AR"][fold_num][target] = ar_df
            print(f"  → {len(ar_df)} predictions (order={ar_order}, seasonal={ar_seasonal})")

            print("\n[ARIMA]")
            arima_order, arima_seasonal = arima_runner.run(splits, target=target, fold=fold_idx)
            arima_df = arima_runner.predict(splits, target=target, fold=fold_idx, step=args.horizon)
            results["ARIMA"][fold_num][target] = arima_df
            print(f"  → {len(arima_df)} predictions (order={arima_order}, seasonal={arima_seasonal})")

            print("\n[TFT – training]")
            ckpt = tft_runner.run(
                splits, target=target, fold=fold_idx,
                use_wandb=args.wandb, device=args.device,
            )
            print(f"  → best checkpoint: {ckpt}")

            # print("\n[TFT – interpretation]")
            # interpretation = tft_runner.interpret_output()
            # print(f"interpretation: {interpretation}")

            print("\n[TFT – inference]")
            tft_df = tft_runner.predict(
                ckpt, splits, target=target, fold=fold_idx, device=args.device,
                step=args.horizon, # includes horizon
            )
            results["TFT"][fold_num][target] = tft_df
            print(f"  → {len(tft_df)} predictions")

    # ── 3. save predictions vs. actuals, calculate eval metrics ──────────────
    save_results(results, out_dir)

    # ── 4. plot results ───────────────────────────────────────────────────────
    plot_results(results, out_dir)


if __name__ == "__main__":
    main()
