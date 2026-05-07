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
from data_frame_builder import DataFrameBuilder
from benchmark_runner    import ARRunner, ARIMARunner
from tft_runner          import TFTRunner

MAIN_TARGETS = ["CPI", "UNRATE", "GDP"]
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

# TODO: change here to sort by target
def save_results(results: dict, out_dir: Path) -> pd.DataFrame:
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for model, model_res in results.items():
        for target, df in model_res.items():
            df.to_csv(out_dir / f"{model.lower()}_{target.lower()}_predictions.csv", index=False)
            m = compute_metrics(df)
            rows.append({"model": model, "target": target, **m})

    metrics_df = pd.DataFrame(rows)[["model", "target", "MAE", "RMSE", "MAPE"]]
    metrics_df = metrics_df.sort_values(by=["target", "model"]).reset_index(drop=True)
    metrics_df.to_csv(out_dir / "metrics.csv", index=False)
    print("\n=== Evaluation Metrics ===")
    print(metrics_df.to_string(index=False))
    return metrics_df


# ── plot ──────────────────────────────────────────────────────────────────────

def plot_results(results: dict, out_dir: Path):
    targets = sorted({t for res in results.values() for t in res})
    _, axes = plt.subplots(len(targets), 1, figsize=(14, 4 * len(targets)), squeeze=False)

    colors     = {"AR": "crimson", "ARIMA": "darkorange", "TFT": "steelblue"}
    linestyles = {"AR": "--",      "ARIMA": "-.",          "TFT": ":"}

    for i, target in enumerate(targets):
        ax = axes[i][0]
        actual_drawn = False

        for model, model_res in results.items():
            if target not in model_res:
                continue
            df = model_res[target].dropna(subset=["actual", "predicted"])

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
        "--embedding", default="fomc-roberta", choices=["fomc-roberta", "finbert"], # added finbert here
        help="Speech embedding to include (default: none — macro-only mode)",
    )
    args = parser.parse_args()

    root    = Path(__file__).parent.parent.parent
    out_dir = root / "out" / "holdout"

    if args.wandb:
        _setup_wandb()

    print(f"Targets   : {args.targets}")
    print(f"Device    : {args.device}")
    print(f"Embedding : {args.embedding or 'none'}")
    print(f"W&B       : {args.wandb}")
    print(f"Output    : {out_dir}\n")

    # ── 1. split the data acc. to data-frame-builder ─────────────────────────
    dfb = DataFrameBuilder(str(root), embedding=args.embedding)
    df  = dfb.process_data()
    splits, holdout = dfb.generate_split(df)

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
    # initiate both runners
    ar_runner  = ARRunner(dfb)
    arima_runner  = ARIMARunner(dfb)
    tft_runner = TFTRunner(dfb)

    results: dict[str, dict[str, pd.DataFrame]] = {"AR": {}, "ARIMA": {}, "TFT": {}}

    # ── 2. let benchmark and tft run ─────────────────────────────────────────
    for target in args.targets:
        print(f"\n{'─' * 55}")
        print(f" Target: {target}")
        print(f"{'─' * 55}")

        print("\n[AR(1)]")
        ar_runner.run(splits, target=target, fold=0)
        ar_df = ar_runner.predict(splits, target=target, fold=0)
        results["AR"][target] = ar_df
        print(f"  → {len(ar_df)} predictions")


        # add arima order for each variable
        print("\n[ARIMA]")
        arima_order, arima_seasonal = arima_runner.run(splits, target=target, fold=0)
        arima_df = arima_runner.predict(splits, target=target, fold=0)
        results["ARIMA"][target] = arima_df
        print(f"  → {len(arima_df)} predictions (order={arima_order}, seasonal={arima_seasonal})")

        print("\n[TFT – training]")
        ckpt = tft_runner.run(
            splits, target=target, fold=0,
            use_wandb=args.wandb, device=args.device,
        )
        print(f"  → best checkpoint: {ckpt}")

        print("\n[TFT – inference]")
        tft_df = tft_runner.predict(
            ckpt, splits, target=target, fold=0, device=args.device,
        )
        results["TFT"][target] = tft_df
        print(f"  → {len(tft_df)} predictions")

    # ── 3. save predictions vs. actuals, calculate eval metrics ──────────────
    save_results(results, out_dir)

    # ── 4. plot results ───────────────────────────────────────────────────────
    plot_results(results, out_dir)


if __name__ == "__main__":
    main()
