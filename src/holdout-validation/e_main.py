#!/usr/bin/env python3
"""
Holdout validation pipeline: AR(1) benchmark vs TFT.

Usage:
    python src/holdout-validation/e_main.py
    python src/holdout-validation/e_main.py --targets CPI GDP --device mps --wandb
    python src/holdout-validation/e_main.py --targets UNRATE --device cuda
"""
import argparse
import json
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
from b_data_frame_builder import DataFrameBuilder
from c_benchmark_runner   import ARRunner, ARIMARunner
from d_tft_runner         import TFTRunner
from a_embedding_manager  import EmbeddingManager

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


# ── tuned hparams ────────────────────────────────────────────────────────────

def load_tuned_hparams(root: Path, target: str, embedding: str | None, horizon: int) -> tuple[dict, dict, str | None]:
    """Load best params saved by hyperparameter tuning.

    Returns (arch_params, emb_params, embedding) — params filtered of '_' keys.
    arch_params: always from macro_only_h{horizon}/best_params.json (target-specific TFT arch)
    emb_params:  from {embedding}_h{horizon}/best_params.json, or {} if embedding is None

    embedding == "auto": scan all tuning dirs for target+horizon, pick the one with lowest _best_mae
    embedding == None:   macro-only mode, no emb_params
    embedding == <name>: load params for that specific embedding
    """
    tuning_dir = root / "out" / "tuning" / target
    suffix     = f"_h{horizon}"

    def _read_raw(path: Path) -> dict:
        if path.exists():
            return json.loads(path.read_text())
        print(f"  [tuned] WARNING: {path} not found — using defaults")
        return {}

    def _strip_private(d: dict) -> dict:
        return {k: v for k, v in d.items() if not k.startswith("_")}

    # auto-detect best embedding from tuning dirs when --embedding passed without a value
    if embedding == "auto":
        macro_raw = _read_raw(tuning_dir / f"macro_only{suffix}" / "best_params.json")
        best_mae  = macro_raw.get("_best_mae", float("inf"))
        best_emb  = None
        if tuning_dir.exists():
            for d in sorted(tuning_dir.iterdir()):  # sorted for determinism
                if d.is_dir() and d.name.endswith(suffix) and not d.name.startswith("macro_only"):
                    raw = _read_raw(d / "best_params.json")
                    if raw.get("_best_mae", float("inf")) < best_mae:
                        best_mae = raw["_best_mae"]
                        best_emb = d.name[: -len(suffix)]  # strip _h{horizon} suffix
        embedding = best_emb
        label = f"'{embedding}' (MAE={best_mae:.6f})" if embedding else f"macro-only (MAE={best_mae:.6f})"
        print(f"  [tuned] auto-selected embedding: {label}")

    # macro-only mode: arch params only, no emb_params
    arch_params = _strip_private(_read_raw(tuning_dir / f"macro_only{suffix}" / "best_params.json"))
    emb_params  = _strip_private(_read_raw(tuning_dir / f"{embedding}{suffix}" / "best_params.json")) if embedding else {}
    return arch_params, emb_params, embedding


# ── metrics ──────────────────────────────────────────────────────────────────

def compute_metrics(df: pd.DataFrame) -> dict:
    errors = df["actual"].values - df["predicted"].values
    mae  = float(np.mean(np.abs(errors)))
    rmse = float(np.sqrt(np.mean(errors ** 2)))

    # Relative RMSE (divide RMSE by std of actuals)
    # std_actual = np.std(df["actual"].values)
    # rel_rmse = float(rmse / std_actual) if std_actual > 0 else np.nan

    # Quantile loss (if quantiles available)
    quantile_loss = np.nan
    if "pred_lo" in df.columns and "pred_hi" in df.columns:
        # Use 0.1 and 0.9 quantiles if present, else fallback to 0.05/0.95
        q_lo = "pred_lo"
        q_hi = "pred_hi"

        # Compute pinball loss for both quantiles and average
        def pinball_loss(y, y_hat, q):
            delta = y - y_hat
            return np.mean(np.maximum(q * delta, (q - 1) * delta))
        y = df["actual"].values
        ql_10 = pinball_loss(y, df[q_lo].values, 0.1)
        ql_90 = pinball_loss(y, df[q_hi].values, 0.9)
        quantile_loss = float((ql_10 + ql_90) / 2)

    return {
        # raw
        "MAE": mae,           
        "RMSE": rmse,
        # "rel_RMSE": rel_rmse,
        "QuantileLoss": quantile_loss if not np.isnan(quantile_loss) else np.nan,
        # rounded versions for display
        "MAE_display": round(mae, 5),
        "RMSE_display": round(rmse, 5),
        # "rel_RMSE_display": round(rel_rmse, 5),
        "QuantileLoss_display": round(quantile_loss, 5) if not np.isnan(quantile_loss) else np.nan,
    }


# ── save ─────────────────────────────────────────────────────────────────────

def save_results(results: dict, out_dir: Path, run_name="default", embedding="none", aggregation="mean", ablation_mode=False, horizon: int = 12) -> pd.DataFrame:
    """Save per-fold CSVs + per-fold metrics + fold-averaged metrics.

    ablation_mode=True  → append to out/holdout/experiments.csv (multi-run comparison)
    ablation_mode=False → overwrite out/holdout/{run_name}/metrics_h{horizon}_{embedding}.csv
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    run_dir = out_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    
    rows = []
    for model, fold_res in results.items():
        for fold, target_res in fold_res.items():
            for target, df in target_res.items():
                df.to_csv(run_dir  / f"{model.lower()}_{target.lower()}_fold{fold}_predictions.csv", index=False)
                m = compute_metrics(df)
                rows.append({"model": model, "fold": fold, "target": target, **m})

    metrics_df = pd.DataFrame(rows)[[c for c in ["model", "fold", "target", "MAE_display", "RMSE_display", 
                                                #  "rel_RMSE_display", 
                                                 "QuantileLoss_display"] if c in pd.DataFrame(rows).columns]]
    metrics_df = metrics_df.sort_values(by=["target", "model", "fold"]).reset_index(drop=True)
    metrics_df = metrics_df.rename(columns={
        "MAE_display": "MAE",
        "RMSE_display": "RMSE", 
        # "rel_RMSE_display": "rel_RMSE",
        "QuantileLoss_display": "QuantileLoss"
    })
    metrics_df.to_csv(run_dir  / "metrics_per_fold.csv", index=False)

    avg_df = (
        metrics_df.groupby(["model", "target"])[[c for c in ["MAE", "RMSE",
                                                            #   "rel_RMSE", 
                                                              "QuantileLoss"] if c in metrics_df.columns]]
        .mean().round(5).reset_index()
        .sort_values(by=["target", "model"]).reset_index(drop=True)
    )
    # append to master experiments file
    avg_df["run_name"]    = run_name
    avg_df["embedding"]   = embedding
    avg_df["aggregation"] = aggregation
    avg_df["timestamp"]   = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M")
    
    if ablation_mode:
        # ablation run: append to shared experiments.csv for multi-run comparison
        master_path = out_dir / "experiments.csv"
        avg_df.to_csv(master_path, mode="a", header=not master_path.exists(), index=False)
    else:
        # standalone run: overwrite per-run metrics file (clean slate each time)
        emb_tag = embedding if embedding != "none" else "macro"
        metrics_path = run_dir / f"metrics_h{horizon}_{emb_tag}.csv"
        avg_df.to_csv(metrics_path, index=False)
        print(f"Metrics saved → {metrics_path}")

    print("\n=== Evaluation Metrics (per fold) ===")
    print(metrics_df.to_string(index=False))
    print("\n=== Evaluation Metrics (averaged across folds) ===")
    print(avg_df.to_string(index=False))
    return avg_df


# ── plot ──────────────────────────────────────────────────────────────────────

def plot_results(results: dict, out_dir: Path, run_name: str = "default", horizon: int = 12, embedding: str | None = None):
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

            # Add prediction interval fill for TFT quantiles if available
            if model == "TFT" and "pred_lo" in df.columns and "pred_hi" in df.columns:
                ax.fill_between(df["date"], df["pred_lo"], df["pred_hi"],
                    color=colors["TFT"], alpha=0.15, label="TFT 10–90% PI")

        ax.set_title(f"{target}: Predicted vs Actual")
        ax.set_xlabel("Date")
        ax.legend()
        ax.grid(alpha=0.3)

    plt.tight_layout()
    run_dir = out_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    emb_tag = embedding if embedding else "macro"
    path = run_dir / f"predictions_vs_actuals_h{horizon}_{emb_tag}.png"
    plt.savefig(path, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"Plot saved → {path}")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    # parse arguments passed by user
    parser = argparse.ArgumentParser(
        description="Holdout validation pipeline: AR(1) benchmark vs TFT"
    )
    
    # add a parser so that we do not need to rerun baselines all the time haha
    parser.add_argument(
        "--no-baselines", action="store_true", default=False,
        help="Skip AR and ARIMA baselines, only run TFT",
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
        "--embedding", nargs="?", const="auto", default=None,
        choices=["auto", "fomc-roberta", "finbert", "finbert-kafka", "fomc-roberta-kafka", "roberta", "llama3.1"],
        help=(
            "Speech embedding: omit for macro-only; pass flag alone (--embedding) to "
            "auto-select best from tuning dirs; or give a name (--embedding fomc-roberta)"
        ),
    )
    
    parser.add_argument(
        "--aggregation", default="mean", choices=["mean", "decay", "attention", "context_attention"],
        help="Speech aggregation strategy (default: mean)",
    )
    
    parser.add_argument(
        "--horizon", type=int, default=12, choices=[3, 6, 12],
        help="Forecast horizon in months (default: 12)",
    )
    
    # add for different types of dimensionality reduction
    parser.add_argument(
        "--reduction", default="pca", choices=["pca", "fa", "none"],
        help="Dimensionality reduction strategy (default: pca)",
    )
    parser.add_argument(
        "--n-pca", type=int, default=5,
        help="Number of PCA components (default: 5, only used when --reduction pca)",
    )
    
    # to save runs individually
    parser.add_argument(
        "--run-name", default="default",
        help="Unique name for this run (used for output subfolder)",
    )

    parser.add_argument(
        "--ablation-mode", action="store_true", default=False,
        help="Append results to experiments.csv (multi-run); default overwrites per-run metrics.csv",
    )
    parser.add_argument(
        "--tuned", action="store_true", default=False,
        help=(
            "Load best hyperparams from out/tuning/{TARGET}/macro_only_h{horizon}/ "
            "(arch) and out/tuning/{TARGET}/{embedding}_h{horizon}/ (emb). "
            "Embedding params are taken from the first target; arch params are per-target."
        ),
    )

    # ── TFT hyperparams (override defaults from HPARAMS_DEFAULTS) ────────────
    parser.add_argument("--encoder-length",         type=int,   default=24,   help="max_encoder_length (default 24)")
    parser.add_argument("--speech-window",          type=int,   default=12,   help="SPEECH_WINDOW_MONTHS (default 12)")
    parser.add_argument("--lstm-layers",            type=int,   default=4,    help="TFT lstm_layers (default 4)")
    parser.add_argument("--hidden-size",            type=int,   default=64,   help="TFT hidden_size (default 64)")
    parser.add_argument("--hidden-continuous-size", type=int,   default=8,    help="TFT hidden_continuous_size (default 8)")
    parser.add_argument("--dropout",                type=float, default=0.2,  help="TFT dropout (default 0.2)")
    parser.add_argument("--lr",                     type=float, default=0.03, help="learning rate (default 0.03)")
    parser.add_argument(
        "--normalizer", default="encoder_none", choices=["encoder_none", "group"],
        help="Target normalizer: encoder_none = EncoderNormalizer(None), group = GroupNormalizer (default: encoder_none)",
    )
    
    args = parser.parse_args()

    # make sure that there is no sneaky cuda being used 
    if args.device != "cuda":                                                                                                                                            
          os.environ["CUDA_VISIBLE_DEVICES"] = ""   

    root    = Path(__file__).parent.parent.parent
    out_dir = root / "out" / "holdout"

    if args.wandb:
        _setup_wandb()

    # ── apply tuned embedding params (from first target) before data loading ──
    # embedding type is auto-detected from best_mae if not explicitly given
    if args.tuned:
        _, emb_params, resolved_emb = load_tuned_hparams(
            root, args.targets[0], args.embedding, args.horizon
        )
        args.embedding = resolved_emb  # may stay None (macro-only wins) or be auto-set
        if emb_params:
            args.aggregation   = emb_params.get("aggregation",          args.aggregation)
            args.reduction     = emb_params.get("reduction",            args.reduction)
            args.n_pca         = emb_params.get("n_pca",                args.n_pca)
            args.speech_window = emb_params.get("speech_window_months", args.speech_window)
            print(f"[tuned] embedding params from '{args.targets[0]}': {emb_params}")
        if len(args.targets) > 1:
            print(f"[tuned] NOTE: embedding/arch params for first target '{args.targets[0]}' guide data loading")

    print(f"Targets   : {args.targets}")
    print(f"Device    : {args.device}")
    print(f"Embedding : {args.embedding or 'none'}")
    print(f"Horizon   : {args.horizon}")
    print(f"Tuned     : {args.tuned}")
    print(f"W&B       : {args.wandb}")
    print(f"Output    : {out_dir}\n")

    # ── 1. split the data acc. to data-frame-builder ─────────────────────────
    # dissents are loaded explicitly so those features appear in process_data()
    # regardless of whether embeddings are used
    hparams = {
        "max_prediction_length":  args.horizon,
        "max_encoder_length":     args.encoder_length,
        "speech_window_months":   args.speech_window,
        "lstm_layers":            args.lstm_layers,
        "hidden_size":            args.hidden_size,
        "hidden_continuous_size": args.hidden_continuous_size,
        "dropout":                args.dropout,
        "learning_rate":          args.lr,
        "normalizer":             args.normalizer,
    }

    dfb = DataFrameBuilder(str(root), aggregation=args.aggregation, speech_window=args.speech_window)
    dfb.load_fomc_dissent()
    df  = dfb.process_data()
    splits, holdout = dfb.generate_split(df)

    if args.embedding is not None: # allow for different models!
        # re-fit PCA per fold on training speeches only — no look-ahead leakage
        # NOW: depending on whether we actually call pca or "none" (or factor analysis, to do)
        emb = EmbeddingManager(
            str(root), 
            embedding=args.embedding, 
            n_pca=args.n_pca,
            reduction=args.reduction
        ).load()
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
    # initiate baseline runners (stateless between targets)
    ar_runner    = ARRunner(dfb,    max_prediction_length=args.horizon)
    arima_runner = ARIMARunner(dfb, max_prediction_length=args.horizon)

    # results[model][fold_num][target] = predictions DataFrame
    results = {"TFT": {s["fold"]: {} for s in splits}}
    if not args.no_baselines:
        results["AR"]    = {s["fold"]: {} for s in splits}
        results["ARIMA"] = {s["fold"]: {} for s in splits}

    # ── 2. run all models for each target × fold ──────────────────────────────
    # outer loop over targets so we can load per-target tuned arch hparams once
    for target in args.targets:
        # load per-target tuned architecture hparams if --tuned
        if args.tuned:
            arch_params, _, _ = load_tuned_hparams(root, target, args.embedding, args.horizon)
            target_hparams = {**hparams, **arch_params}
            print(f"\n[tuned] {target} arch params: {arch_params}")
        else:
            target_hparams = hparams
        tft_runner = TFTRunner(dfb, hparams=target_hparams)

        for fold_idx, s in enumerate(splits):
            fold_num = s["fold"]
            print(f"\n{'─' * 55}")
            print(f" Fold {fold_num} | Target: {target}")
            print(f"{'─' * 55}")

            if not args.no_baselines:
                print("\n[AR]")
                ar_order, ar_seasonal = ar_runner.run(splits, target=target, fold=fold_idx)
                ar_df = ar_runner.predict(splits, target=target, fold=fold_idx)
                results["AR"][fold_num][target] = ar_df
                print(f"  → {len(ar_df)} predictions (order={ar_order}, seasonal={ar_seasonal})")

                print("\n[ARIMA]")
                arima_order, arima_seasonal = arima_runner.run(splits, target=target, fold=fold_idx)
                arima_df = arima_runner.predict(splits, target=target, fold=fold_idx)
                results["ARIMA"][fold_num][target] = arima_df
                print(f"  → {len(arima_df)} predictions (order={arima_order}, seasonal={arima_seasonal})")

            print("\n[TFT – training]")
            ckpt = tft_runner.run(
                splits, target=target, fold=fold_idx,
                use_wandb=args.wandb, device=args.device,
                run_name=f"{args.run_name}-{target}-fold{fold_num}",
            )
            print(f"  → best checkpoint: {ckpt}")

            # print("\n[TFT – interpretation]")
            # interpretation = tft_runner.interpret_output()
            # print(f"interpretation: {interpretation}")

            print("\n[TFT – inference]")
            tft_df = tft_runner.predict(
                ckpt, splits, target=target, fold=fold_idx, device=args.device,
            )
            results["TFT"][fold_num][target] = tft_df
            print(f"  → {len(tft_df)} predictions")

    # ── 3. save predictions vs. actuals, calculate eval metrics ──────────────
    save_results(results, out_dir,
             run_name=args.run_name,
             embedding=args.embedding or "none",
             aggregation=args.aggregation,
             ablation_mode=args.ablation_mode,
             horizon=args.horizon)

    # ── 4. plot results ───────────────────────────────────────────────────────
    plot_results(results, out_dir, run_name=args.run_name,
                 horizon=args.horizon, embedding=args.embedding)


if __name__ == "__main__":
    main()
