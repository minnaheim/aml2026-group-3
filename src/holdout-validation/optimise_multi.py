"""
Multi-target TFT hyperparameter optimisation via Optuna (Pareto front).

Usage:
    python optimise_multi.py
    python optimise_multi.py --embedding fomc-roberta --aggregation mean
    python optimise_multi.py --embedding finbert --aggregation decay --seed 0

Arguments:
    --embedding     Speech embedding to include. Optional. One of:
                        fomc-roberta, finbert, finbert-kafka, fomc-roberta-kafka, roberta.
                    Omit for macro-only mode (no speech features).
    --aggregation   How speeches within a window are aggregated into a single vector.
                    One of: mean, decay, attention, context_attention. Default: mean.
    --seed          Random seed for Optuna TPE sampler. Default: 42.
    --horizon       Forecast horizon in months to optimise for. Default: 12.

Output:
    Pareto-front trials are appended to:
        out/tft/optimisation/optuna_pareto_front_joint.csv
"""
import argparse
import copy
import json
import lightning.pytorch as pl
import optuna
import os
import pandas as pd
import time

from lightning.pytorch.callbacks import EarlyStopping
from pytorch_forecasting import TemporalFusionTransformer
from pytorch_forecasting.metrics import QuantileLoss
from tft_runner import TFTRunner
from embedding_manager import EmbeddingManager
from data_frame_builder import DataFrameBuilder

N_TRIALS = 3
TARGET_VARIABLES = ["UNRATE", "CPI"]

def prepare_trial_data(trial_dfb, speech_window, n_pca, embedding_name, project_path):
    """Handles data reprocessing and leakage-free embedding reconstruction per trial."""
    df = trial_dfb.process_data()
    splits, _ = trial_dfb.generate_split(df)

    if embedding_name is not None:
        if hasattr(trial_dfb, 'SPEECH_WINDOW_MONTHS'):
            trial_dfb.SPEECH_WINDOW_MONTHS = speech_window
        # n_pca is passed here so PCA is fitted with the correct number of components for this trial
        emb_mgr = EmbeddingManager(project_path, embedding=embedding_name, n_pca=n_pca).load()
        splits = trial_dfb.add_leakage_free_embeddings(splits, emb_mgr)
        # Slice PCA feature columns safely within this trial instance
        if hasattr(trial_dfb, 'pca_cols'):
            trial_dfb.pca_cols = trial_dfb.pca_cols[:n_pca]

    return splits


def train_tft(runner, training_ds, train_dl, val_dl, hparams):
    """Isolates the network configuration and PyTorch Lightning execution loop."""
    # Note: PyTorchLightningPruningCallback is not used — multi-objective pruning is unsupported
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=runner.PATIENCE, mode='min'),
    ]

    trainer = pl.Trainer(
        max_epochs=runner.MAX_EPOCHS,
        accelerator="auto",
        devices=1,
        gradient_clip_val=0.25,
        callbacks=callbacks,
        enable_model_summary=False,
        logger=False
    )

    tft = TemporalFusionTransformer.from_dataset(
        training_ds,
        learning_rate=hparams["learning_rate"],
        lstm_layers=hparams["lstm_layers"],
        hidden_size=hparams["hidden_size"],
        attention_head_size=hparams["attention_head_size"],
        dropout=hparams["dropout"],
        hidden_continuous_size=hparams["hidden_continuous_size"],
        output_size=len(runner.QUANTILES),
        loss=QuantileLoss(quantiles=runner.QUANTILES),
        reduce_on_plateau_patience=3,
    )

    trainer.fit(tft, train_dataloaders=train_dl, val_dataloaders=val_dl)
    return trainer.callback_metrics["val_loss"].item()


def objective(trial, dfb_master, embedding_name, project_path, horizon):
    """Optuna objective function managing clean parameter sampling and pipeline coordination."""

    # 1. Parameter Sampling Spaces Matrix
    hparams = {
        "max_encoder_length": trial.suggest_categorical("max_encoder_length", [12, 24, 36, 48]),
        "hidden_size": trial.suggest_categorical("hidden_size", [16, 32, 64, 128]),
        "lstm_layers": trial.suggest_categorical("lstm_layers", [1, 2, 4]),
        "attention_head_size": trial.suggest_categorical("attention_head_size", [1, 2, 4]),
        "hidden_continuous_size": trial.suggest_categorical("hidden_continuous_size", [8, 16, 32]),
        "dropout": trial.suggest_float("dropout", 0.05, 0.4),
        "learning_rate": trial.suggest_float("learning_rate", 1e-4, 0.1, log=True),
        "batch_size": trial.suggest_categorical("batch_size", [16, 32, 64, 128]),
    }
    # Only tune embedding-specific params when an embedding is actually used
    if embedding_name is not None:
        hparams["n_pca"] = trial.suggest_categorical("n_pca", [10, 20, 30])
        hparams["speech_window_months"] = trial.suggest_categorical("speech_window_months", [3, 6, 12])

    # Hidden size must be divisible by attention head count
    if hparams["hidden_size"] % hparams["attention_head_size"] != 0:
        raise optuna.exceptions.TrialPruned("Invalid structural dimensions: hidden_size must be divisible by attention_head_size.")

    # 2. Shared data preparation once per trial (embeddings are target-agnostic)
    dfb_trial = copy.deepcopy(dfb_master)
    splits = prepare_trial_data(
        dfb_trial,
        hparams.get("speech_window_months"),
        hparams.get("n_pca"),
        embedding_name,
        project_path
    )

    fold_to_tune_on = 2  # Fold 3 Evaluation
    target_losses = []

    # 3. Train one model per target, accumulate losses
    for target in TARGET_VARIABLES:
        runner = TFTRunner(dfb_trial)
        runner.LEARNING_RATE = hparams["learning_rate"]
        runner.MAX_ENCODER_LENGTH = hparams["max_encoder_length"]
        runner.MAX_PREDICTION_LENGTH = horizon
        runner.MAX_EPOCHS = 30
        runner.PATIENCE = 7

        train_raw = dfb_trial.get_data(splits, train=True, model='TFT', fold=fold_to_tune_on)
        train_df = runner._add_tft_vars(train_raw, target)

        training_ds, train_dl, val_dl = runner.create_tft_dataset(
            train_df, target, fold_to_tune_on, batch_size=hparams["batch_size"]
        )

        loss_val = train_tft(runner, training_ds, train_dl, val_dl, hparams)
        target_losses.append(loss_val)

    # Return a tuple of losses: (UNRATE_loss, CPI_loss, GDP_loss)
    return tuple(target_losses)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-target hyperparameter optimization for TFT using Optuna")
    parser.add_argument("--embedding", default=None, choices=["fomc-roberta", "finbert", "finbert-kafka", "fomc-roberta-kafka", "roberta"])
    parser.add_argument("--aggregation", default="mean", choices=["mean", "decay", "attention", "context_attention"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--horizon", type=int, default=12, choices=[1, 3, 6, 9, 12], help="Forecast horizon in months (default: 12)")
    args = parser.parse_args()

    project_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    master_engine = DataFrameBuilder(project_path, aggregation=args.aggregation)

    if args.embedding is not None:
        df = master_engine.process_data()
        splits, _ = master_engine.generate_split(df)
        # Initialise with max n_pca=30; individual trials will slice down as needed
        emb_mgr = EmbeddingManager(project_path, embedding=args.embedding, n_pca=30).load()
        splits = master_engine.add_leakage_free_embeddings(splits, emb_mgr)

    # Create Multi-Objective Study
    study = optuna.create_study(
        directions=["minimize" for _ in TARGET_VARIABLES],
        sampler=optuna.samplers.TPESampler(seed=args.seed)
    )

    print(f"Starting multi-target hyperparameter optimization search across: {TARGET_VARIABLES}")
    t_start = time.time()
    study.optimize(
        lambda trial: objective(trial, master_engine, args.embedding, project_path, args.horizon),
        n_trials=N_TRIALS
    )
    total_minutes = (time.time() - t_start) / 60

    print("\n=== Multi-Objective Optimization Complete ===")
    print(f"Number of trials on the Pareto Front: {len(study.best_trials)}")

    out_dir = os.path.join(project_path, "out", "tft", "optimisation")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "optuna_pareto_front_joint.csv")

    pareto_rows = []
    for i, trial in enumerate(study.best_trials):
        loss_str = " | ".join(f"{t}: {v:.5f}" for t, v in zip(TARGET_VARIABLES, trial.values))
        print(f"\n--- Pareto Frontier Trial {i+1} (Trial #{trial.number}) ---")
        print(f"  Losses -> {loss_str}")

        # Build row in fixed column order so schema is consistent across embedding/macro-only runs
        row_dict = {
            "max_encoder_length": trial.params.get("max_encoder_length"),
            "n_pca": trial.params.get("n_pca"),
            "speech_window_months": trial.params.get("speech_window_months"),
            "hidden_size": trial.params.get("hidden_size"),
            "lstm_layers": trial.params.get("lstm_layers"),
            "attention_head_size": trial.params.get("attention_head_size"),
            "hidden_continuous_size": trial.params.get("hidden_continuous_size"),
            "dropout": trial.params.get("dropout"),
            "learning_rate": trial.params.get("learning_rate"),
            "batch_size": trial.params.get("batch_size"),
        }
        for target, val in zip(TARGET_VARIABLES, trial.values):
            row_dict[f"{target}_val_loss"] = val
        row_dict["Trial_Number"] = trial.number
        row_dict["Embedding"] = args.embedding
        row_dict["Aggregation"] = args.aggregation
        row_dict["Horizon"] = args.horizon
        row_dict["Seed"] = args.seed
        pareto_rows.append(row_dict)

    # Export the whole frontier to look for compromise candidates
    # Write headers when the file doesn't exist or is empty (guards against zero-byte files from prior runs)
    needs_header = not (os.path.exists(out_path) and os.path.getsize(out_path) > 0)
    df_pareto = pd.DataFrame(pareto_rows)
    df_pareto.to_csv(out_path, mode='a', index=False, header=needs_header)
    print(f"\nPareto front matrix appended to: {out_path}")

    # Also save a JSON mirroring the per-trial structure of best_params.json (one entry per Pareto trial)
    json_rows = []
    for row in pareto_rows:
        entry = {k: v for k, v in row.items() if not k.endswith("_val_loss") and k not in ("Trial_Number", "Embedding", "Aggregation", "Horizon", "Seed")}
        for target in TARGET_VARIABLES:
            entry[f"_{target}_val_loss"] = row[f"{target}_val_loss"]
        entry["_trial_number"] = row["Trial_Number"]
        entry["_n_trials"] = N_TRIALS
        entry["_total_minutes"] = round(total_minutes, 1)
        entry["_minutes_per_trial"] = round(total_minutes / N_TRIALS, 1)
        entry["embedding"] = args.embedding
        entry["aggregation"] = args.aggregation
        entry["horizon"] = args.horizon
        entry["seed"] = args.seed
        json_rows.append(entry)

    # Append to existing JSON list (mirrors the CSV append behaviour)
    json_path = os.path.join(out_dir, "optuna_pareto_front_joint.json")
    if os.path.exists(json_path) and os.path.getsize(json_path) > 0:
        with open(json_path, "r") as f:
            existing = json.load(f)
        json_rows = existing + json_rows
    with open(json_path, "w") as f:
        json.dump(json_rows, f, indent=4)
    print(f"Pareto front JSON appended to: {json_path}")

    # --- Optuna Visualizations (saved as interactive HTML) ---
    # Multi-objective studies require a `target` callable to extract one scalar per trial.
    # The default `i=i` capture prevents the classic closure-over-loop-variable bug.
    plots_dir = os.path.join(out_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    for i, tgt in enumerate(TARGET_VARIABLES):
        target_fn = lambda t, i=i: t.values[i]
        label = f"{tgt}_val_loss"

        optuna.visualization.plot_optimization_history(study, target=target_fn, target_name=label).write_html(
            os.path.join(plots_dir, f"optimization_history_{tgt}.html")
        )
        optuna.visualization.plot_parallel_coordinate(study, target=target_fn, target_name=label).write_html(
            os.path.join(plots_dir, f"parallel_coordinate_{tgt}.html")
        )
        optuna.visualization.plot_slice(study, target=target_fn, target_name=label).write_html(
            os.path.join(plots_dir, f"slice_{tgt}.html")
        )
        optuna.visualization.plot_param_importances(study, target=target_fn, target_name=label).write_html(
            os.path.join(plots_dir, f"param_importances_{tgt}.html")
        )

    # Pareto-front scatter — native multi-objective plot, no target callable needed
    optuna.visualization.plot_pareto_front(
        study, target_names=[f"{t}_val_loss" for t in TARGET_VARIABLES]
    ).write_html(os.path.join(plots_dir, "pareto_front.html"))

    print(f"Optuna visualizations saved to: {plots_dir}")