#!/usr/bin/env python3
"""
Bayesian hyperparameter tuning for TFT using Optuna (TPE sampler).

Runs a per-target study; each trial trains TFT on the first --n-folds CV folds
and returns the mean MAE. Best params saved to out/tuning/{target}/best_params.json.

Usage:
    python src/holdout-validation/g_tune_hyperparams.py --target CPI --n-trials 3 --device cpu
    python src/holdout-validation/g_tune_hyperparams.py --target CPI --n-trials 50 --device cuda
    # with embeddings (once runtime is known):
    python src/holdout-validation/g_tune_hyperparams.py --target CPI --n-trials 50 --embedding fomc-roberta --device cuda
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
def objective(trial: optuna.Trial, args: argparse.Namespace, root: Path, use_wandb: bool = True) -> float:
    max_prediction_length = trial.suggest_categorical("max_prediction_length", [3, 6, 12])
    hparams = {
        # data / feature params
        "max_encoder_length":     trial.suggest_int("max_encoder_length", 12, 48),
        "max_prediction_length":  max_prediction_length,
        "speech_window_months":   trial.suggest_int("speech_window_months", 3, 12),
        "n_pca":                  trial.suggest_int("n_pca", 5, 30),
        # architecture
        "hidden_size":            trial.suggest_categorical("hidden_size", [16, 32, 64, 128]),
        "hidden_continuous_size": trial.suggest_categorical("hidden_continuous_size", [8, 16, 32]),
        "lstm_layers":            trial.suggest_categorical("lstm_layers", [1, 2, 4]),
        "dropout":                trial.suggest_float("dropout", 0.05, 0.4),
        # optimisation
        "learning_rate":          trial.suggest_float("learning_rate", 1e-4, 0.1, log=True),
        # normalizer
        "normalizer":             trial.suggest_categorical("normalizer", ["encoder_none", "group"]),
        # fixed
        "min_encoder_length":     8,
        "attention_head_size":    2,
        "max_epochs":             50,
        "patience":               15,
    }
    # TODO: why define everything separately and let it run here, not in original pipeline?

    speech_window = hparams["speech_window_months"]
    n_pca         = hparams["n_pca"]

    dfb = DataFrameBuilder(str(root), aggregation=args.aggregation, speech_window=speech_window)
    dfb.load_fomc_dissent()
    df = dfb.process_data()
    splits, _ = dfb.generate_split(df) # dont need holdout here

    if args.embedding is not None:
        emb = EmbeddingManager(
            str(root),
            embedding=args.embedding,
            n_pca=n_pca,
            reduction=args.reduction, # type of dim reduction
        ).load()
        splits = dfb.add_leakage_free_embeddings(splits, emb)

    # use only the first n_folds folds to keep each trial fast (set via --n-folds, default 2)
    # defined in --n-folds
    splits = splits[: args.n_folds]

    runner = TFTRunner(dfb, hparams=hparams)
    maes = []

    for fold_idx, s in enumerate(splits):
        try:
            run_name = f"trial-{trial.number}-{args.target}-fold{fold_idx}"
            ckpt = runner.run(
                splits, target=args.target, fold=fold_idx,
                use_tqdm=False, use_wandb=use_wandb, device=args.device,
                run_name=run_name,
            )
            preds = runner.predict(
                ckpt, splits, target=args.target, fold=fold_idx,
                device=args.device, step=max_prediction_length,
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


def main():
    # note: argparse converts hyphens to underscores, so --n-folds → args.n_folds, etc.
    parser = argparse.ArgumentParser(description="Bayesian HP tuning for TFT (Optuna/TPE)")
    parser.add_argument("--target",      required=True, choices=["CPI", "UNRATE", "GDP"])          # → args.target
    parser.add_argument("--n-trials",    type=int,   default=50)                                   # → args.n_trials
    parser.add_argument("--n-folds",     type=int,   default=2,   help="CV folds per trial (default 2 for speed)")  # → args.n_folds
    parser.add_argument("--device",      default="cpu", choices=["cpu", "mps", "cuda"])            # → args.device
    parser.add_argument("--embedding",   default=None,
                        choices=["fomc-roberta", "finbert", "finbert-kafka", "fomc-roberta-kafka"]) # → args.embedding
    parser.add_argument("--aggregation", default="mean",
                        choices=["mean", "decay", "attention", "context_attention"])                # → args.aggregation
    parser.add_argument("--reduction",   default="pca", choices=["pca", "fa", "none"])             # → args.reduction
    parser.add_argument("--wandb", action="store_true", default=False)                             # → args.wandb
    args = parser.parse_args()

    if args.wandb:
        _setup_wandb()

    root    = Path(__file__).parent.parent.parent
    # separate output dir per target + embedding + aggregation + reduction so studies don't overwrite each other
    emb_tag = args.embedding if args.embedding else "macro_only"
    run_tag = f"{emb_tag}_{args.aggregation}_{args.reduction}" if args.embedding else emb_tag
    out_dir = root / "out" / "tuning" / args.target / run_tag
    out_dir.mkdir(parents=True, exist_ok=True)

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
        lambda trial: objective(trial, args, root, use_wandb=args.wandb),
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


if __name__ == "__main__":
    main()
