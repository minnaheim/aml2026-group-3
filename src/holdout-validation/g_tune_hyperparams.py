#!/usr/bin/env python3
"""
Bayesian hyperparameter tuning for TFT using Optuna (TPE sampler).

Two-stage design:
  Stage 1 (no --embedding): tune architecture params (encoder length, hidden size,
      dropout, lr, normalizer) for macro-only TFT.
      Results → out/tuning/{target}/macro_only_h{horizon}/best_params.json

  Stage 2 (--embedding set): fix stage-1 architecture params from JSON, then tune
      speech-embedding params (aggregation, reduction, n_pca, speech_window).
      Results → out/tuning/{target}/{embedding}_h{horizon}/best_params.json

Usage:
    # Stage 1 — macro-only architecture tuning
    python src/holdout-validation/g_tune_hyperparams.py --target CPI --n-trials 100 --horizon 12 --device cpu

    # Stage 2 — speech embedding tuning (single target/horizon)
    python src/holdout-validation/g_tune_hyperparams.py --target CPI --embedding fomc-roberta --n-trials 50 --horizon 12 --device cuda

    # Stage 2 — all targets x horizons in sequence
    python src/holdout-validation/g_tune_hyperparams.py --embedding fomc-roberta --n-trials 50 --all --device cuda
"""
import argparse
import json
import sys
import time
import numpy as np
from pathlib import Path

import optuna
from optuna.samplers import TPESampler

sys.path.insert(0, str(Path(__file__).parent))
from b_data_frame_builder import DataFrameBuilder
from d_tft_runner         import TFTRunner
from a_embedding_manager  import EmbeddingManager
from e_main               import compute_metrics, _setup_wandb

# defines hyperparam grid
def objective(trial: optuna.Trial, args: argparse.Namespace, root: Path, splits, dfb, use_wandb: bool = True) -> float:

    if args.embedding is not None:
        # ── Stage 2: architecture fixed from stage-1 JSON; tune speech params ──
        macro_path = root / "out" / "tuning" / args.target / f"macro_only_h{args.horizon}" / "best_params.json"
        if not macro_path.exists():
            raise FileNotFoundError(
                f"Stage 1 params not found at {macro_path}. Run macro-only tuning first."
            )
        with open(macro_path) as f:
            macro_params = {k: v for k, v in json.load(f).items() if not k.startswith("_")}

        # speech-specific search params
        aggregation   = trial.suggest_categorical("aggregation",        ["mean", "decay", "attention"]) # switch to regular attention since context attention is so slow due to the month loop
        reduction     = trial.suggest_categorical("reduction",          ["pca", "fa"])
        n_pca         = trial.suggest_int("n_pca",                      5, 30)
        speech_window = trial.suggest_int("speech_window_months",       3, 12)

        # architecture fixed from stage 1; lstm_layers not stored in JSON (fixed at 2 in stage 1)
        hparams = {
            "max_prediction_length": args.horizon,
            "min_encoder_length":    8,
            "attention_head_size":   2,
            "max_epochs":            50,
            "patience":              15,
            "lstm_layers":           2,
            **macro_params,
        }

        # rebuild data per trial — aggregation and speech_window both vary
        trial_dfb = DataFrameBuilder(str(root), aggregation=aggregation, speech_window=speech_window, device = args.device)
        trial_dfb.load_fomc_dissent()
        df = trial_dfb.process_data()
        trial_splits, _ = trial_dfb.generate_split(df)
        emb = EmbeddingManager(str(root), embedding=args.embedding, n_pca=n_pca, reduction=reduction).load()
        trial_splits = trial_dfb.add_leakage_free_embeddings(trial_splits, emb)

    else:
        # ── Stage 1: tune architecture params; data precomputed in run_study() ─
        hparams = {
            # data / feature params
            "max_encoder_length":     trial.suggest_int("max_encoder_length", 12, 48),
            "max_prediction_length":  args.horizon,   # fixed — horizon is a research variable, not a tuning param
            # architecture
            "hidden_size":            trial.suggest_categorical("hidden_size", [8, 16, 32, 64, 128, 256]),
            "hidden_continuous_size": trial.suggest_categorical("hidden_continuous_size", [2, 4, 8, 16]),
            "lstm_layers":            2,  # fixed — removed from search (was [4,8,16,32], too slow)
            "dropout":                trial.suggest_float("dropout", 0.05, 0.55),
            # optimisation
            "learning_rate":          trial.suggest_float("learning_rate", 1e-4, 0.15, log=True),
            # normalizer
            "normalizer":             trial.suggest_categorical("normalizer", ["encoder_none", "group"]),  # encoder_softplus excluded: softplus(GDP_levels) overflows float32
            # fixed
            "min_encoder_length":     8,
            "attention_head_size":    2,
            "max_epochs":             50,
            "patience":               15,
        }
        trial_dfb    = dfb      # precomputed once in run_study() — no disk I/O per trial
        trial_splits = splits

    # use only the first n_folds folds to keep each trial fast
    trial_splits = trial_splits[: args.n_folds]

    runner = TFTRunner(trial_dfb, hparams=hparams)
    maes = []

    for fold_idx in range(len(trial_splits)):
        try:
            run_name = f"trial-{trial.number}-{args.target}-fold{fold_idx}"
            ckpt = runner.run(
                trial_splits, target=args.target, fold=fold_idx,
                use_tqdm=False, use_wandb=use_wandb, device=args.device,
                run_name=run_name,
            )
            preds = runner.predict(
                ckpt, trial_splits, target=args.target, fold=fold_idx,
                device=args.device, step=args.horizon,
            )
            m = compute_metrics(preds)
            maes.append(m["MAE"])
        except Exception as e:
            # report a large loss so Optuna deprioritises this region
            print(f"  [trial {trial.number} fold {fold_idx}] failed: {e}")
            maes.append(1e6)
        finally:
            # close each W&B run explicitly so the next trial starts a fresh one
            if use_wandb:
                import wandb
                wandb.finish()

    return float(np.mean(maes))


def run_study(args: argparse.Namespace, root: Path) -> None:
    """Run one Optuna study for args.target + args.horizon."""
    if args.wandb:
        _setup_wandb()

    # output dir: stage 1 → macro_only_h{N}; stage 2 → {embedding}_h{N}
    tag     = args.embedding if args.embedding else "macro_only"
    out_dir = root / "out" / "tuning" / args.target / f"{tag}_h{args.horizon}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # precompute macro-only data once for stage 1 (stage 2 rebuilds per trial)
    if args.embedding is None:
        dfb = DataFrameBuilder(str(root), aggregation=args.aggregation, speech_window=12)
        dfb.load_fomc_dissent()
        df  = dfb.process_data()
        splits, _ = dfb.generate_split(df)
    else:
        dfb, splits = None, None  # stage 2 builds per-trial inside objective()

    study_path = out_dir / "optuna_study.pkl"

    # resume existing study if present
    if study_path.exists():
        import pickle
        with open(study_path, "rb") as f:
            study = pickle.load(f)
        print(f"Resuming study from {study_path} ({len(study.trials)} trials done)")
    else:
        study = optuna.create_study(direction="minimize", sampler=TPESampler(seed=42))

    t0 = time.time()
    study.optimize(
        lambda trial: objective(trial, args, root, splits, dfb, use_wandb=args.wandb),
        n_trials=args.n_trials,
        show_progress_bar=True,
    )
    elapsed      = time.time() - t0
    elapsed_min  = elapsed / 60
    per_trial    = elapsed_min / args.n_trials

    # save study for resuming later
    import pickle
    with open(study_path, "wb") as f:
        pickle.dump(study, f)

    best = study.best_params
    best["_best_mae"]           = study.best_value
    best["_n_trials"]           = args.n_trials
    best["_total_minutes"]      = round(elapsed_min, 1)
    best["_minutes_per_trial"]  = round(per_trial, 1)
    best_path = out_dir / "best_params.json"
    with open(best_path, "w") as f:
        json.dump(best, f, indent=2)

    print(f"\nBest MAE : {study.best_value:.5f}")
    print(f"Runtime  : {elapsed_min:.1f} min total  |  {per_trial:.1f} min/trial  ({args.n_trials} trials)")
    print(f"Best params saved → {best_path}")
    print(json.dumps(best, indent=2))


def main():
    # note: argparse converts hyphens to underscores, so --n-folds → args.n_folds, etc.
    parser = argparse.ArgumentParser(description="Bayesian HP tuning for TFT (Optuna/TPE)")
    parser.add_argument("--target",      default=None, choices=["CPI", "UNRATE", "GDP"],
                        help="Target variable (required unless --all is set)")               # → args.target
    parser.add_argument("--n-trials",    type=int,   default=50)                             # → args.n_trials
    parser.add_argument("--n-folds",     type=int,   default=2,   help="CV folds per trial (default 2 for speed)")  # → args.n_folds
    parser.add_argument("--device",      default="cpu", choices=["cpu", "mps", "cuda"])      # → args.device
    parser.add_argument("--embedding",   default=None,
                        choices=["fomc-roberta", "finbert", "finbert-kafka", "fomc-roberta-kafka"])  # → args.embedding
    parser.add_argument("--aggregation", default="mean",
                        choices=["mean", "decay", "attention", "context_attention"])          # → args.aggregation (stage 1 only)
    parser.add_argument("--reduction",   default="pca", choices=["pca", "fa", "none"])       # → args.reduction (stage 1 only)
    parser.add_argument("--wandb",       action="store_true", default=False)                 # → args.wandb
    parser.add_argument("--horizon",     type=int, default=12, choices=[3, 6, 12],
                        help="Forecast horizon — fixed across all trials (default: 12)")     # → args.horizon
    parser.add_argument("--all",         action="store_true", default=False,
                        help="Stage 2 only: run all targets x horizons sequentially")        # → args.all
    args = parser.parse_args()

    if args.all and args.embedding is None:
        parser.error("--all requires --embedding (stage 2 only)")
    if not args.all and args.target is None:
        parser.error("--target is required unless --all is set")

    root = Path(__file__).parent.parent.parent

    if args.all:
        # run all 9 target × horizon combinations in sequence
        for target in ["CPI", "GDP", "UNRATE"]:
            for horizon in [3, 6, 12]:
                print(f"\n{'='*60}")
                print(f"  Stage 2: {target}  horizon={horizon}  embedding={args.embedding}")
                print(f"{'='*60}")
                args.target  = target
                args.horizon = horizon
                run_study(args, root)
    else:
        run_study(args, root)


if __name__ == "__main__":
    main()
