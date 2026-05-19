"""
Single-target TFT hyperparameter optimisation via Optuna.

Usage:
    python optimise_target.py --target UNRATE
    python optimise_target.py --target CPI --embedding fomc-roberta --aggregation mean
    python optimise_target.py --target GDP --embedding finbert --aggregation decay --seed 0

Arguments:
    --target        Target variable to optimise for. Required. One of: CPI, UNRATE, GDP.
    --embedding     Speech embedding to include. Optional. One of:
                        fomc-roberta, finbert, finbert-kafka, fomc-roberta-kafka, roberta.
                    Omit for macro-only mode (no speech features).
    --aggregation   How speeches within a window are aggregated into a single vector.
                    One of: mean, decay, attention, context_attention. Default: mean.
    --seed          Random seed for Optuna TPE sampler. Default: 42.
    --horizon       Forecast horizon in months to optimise for. Default: 12.

Output:
    Best trial parameters are appended to:
        out/tft/optimisation/optuna_best_params_<target>.csv
"""
import argparse
import copy
import json
import lightning.pytorch as pl
import optuna
import os
import pandas as pd
import time

from pathlib import Path
from lightning.pytorch.callbacks import EarlyStopping
from pytorch_forecasting import TemporalFusionTransformer
from pytorch_forecasting.metrics import QuantileLoss
from optuna.integration import PyTorchLightningPruningCallback
from tft_runner import TFTRunner 
from embedding_manager import EmbeddingManager
from data_frame_builder import DataFrameBuilder

N_TRIALS = 40  

def prepare_trial_data(trial_dfb, speech_window, n_pca, embedding_name, project_path):
    """Handles data reprocessing and leakage-free embedding reconstruction per trial."""
    # Re-process and split data frames
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


def train_tft(runner, training_ds, train_dl, val_dl, hparams, trial):
    """Isolates the network configuration and PyTorch Lightning execution loop."""
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=runner.PATIENCE, mode='min'),
        PyTorchLightningPruningCallback(trial, monitor="val_loss")
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


def objective(trial, dfb_master, target, embedding_name, project_path, horizon):
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
        "batch_size": trial.suggest_categorical("batch_size", [16, 32, 64, 128])
    }
    # only tune embedding-specific params when an embedding is actually used
    if embedding_name is not None:
        hparams["n_pca"] = trial.suggest_categorical("n_pca", [10, 20, 30])
        hparams["speech_window_months"] = trial.suggest_categorical("speech_window_months", [3, 6, 12])

    # Hidden size MUST be perfectly divisible by attention head count
    if hparams["hidden_size"] % hparams["attention_head_size"] != 0:
        raise optuna.exceptions.TrialPruned("Invalid structural dimensions: hidden_size must be divisible by attention_head_size.")

    # 2. Deepcopy and construct dataset vectors safely inside the trial context
    dfb_trial = copy.deepcopy(dfb_master)
    splits = prepare_trial_data(
        dfb_trial,
        hparams.get("speech_window_months"),
        hparams.get("n_pca"),
        embedding_name,
        project_path
    )
        
    fold_to_tune_on = 2  # Fold 3 Evaluation
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

    # 3. Model Training Execution
    return train_tft(runner, training_ds, train_dl, val_dl, hparams, trial)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hyperparameter optimization for TFT using Optuna")
    parser.add_argument("--embedding", default=None, choices=["fomc-roberta", "finbert", "finbert-kafka", "fomc-roberta-kafka", "roberta"])
    parser.add_argument("--target", required=True, choices=["CPI", "UNRATE", "GDP"], help="Target variable to optimize")
    parser.add_argument("--aggregation", default="mean", choices=["mean", "decay", "attention", "context_attention"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--horizon", type=int, default=12, choices=[1, 3, 6, 9, 12], help="Forecast horizon in months (default: 12)")
    args = parser.parse_args()

    project_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    master_engine = DataFrameBuilder(project_path, aggregation=args.aggregation)

    if args.embedding is not None:
        df = master_engine.process_data()
        splits, _ = master_engine.generate_split(df)
        # initialize embedding manager with 30 PCA components by default 
        # individual trials will slice this down as needed
        emb_mgr = EmbeddingManager(project_path, embedding=args.embedding, n_pca=30).load()
        splits = master_engine.add_leakage_free_embeddings(splits, emb_mgr)

    # Create Single-Objective Study
    study = optuna.create_study(
        direction="minimize",  
        sampler=optuna.samplers.TPESampler(seed=args.seed)
    )

    print(f"Starting single-target hyperparameter optimization search for: {args.target}")
    t_start = time.time()
    study.optimize(
        lambda trial: objective(trial, master_engine, args.target, args.embedding, project_path, args.horizon), 
        n_trials=N_TRIALS
    )
    total_minutes = (time.time() - t_start) / 60

    print("\n=== Hyperparameter Optimization Complete ===")
    print(f"Best Trial Validation Loss: {study.best_value:.5f}")
    print("\nOptimized Parameters Matrix:")
    for param_key, param_val in study.best_params.items():
        print(f"  {param_key}: {param_val}")

    # Output storage configurations
    out_dir = os.path.join(project_path, "out", "tft", "optimisation")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"optuna_best_params_{args.target}.csv")

    best_trial_dict = study.best_params.copy()
    best_trial_dict["Best_Validation_Loss"] = study.best_value
    best_trial_dict["Trial_Number"] = study.best_trial.number
    best_trial_dict["Embedding"] = args.embedding
    best_trial_dict["Aggregation"] = args.aggregation
    best_trial_dict["Horizon"] = args.horizon
    best_trial_dict["Seed"] = args.seed

    df_best = pd.DataFrame([best_trial_dict])
    df_best.to_csv(out_path, mode='a', index=False, header=not os.path.exists(out_path))
    print(f"\nBest trial parameters safely appended to: {out_path}")

    # JSON — mirrors best_params.json structure; appends across runs
    json_entry = study.best_params.copy()
    json_entry["_best_val_loss"] = study.best_value
    json_entry["_trial_number"] = study.best_trial.number
    json_entry["_n_trials"] = N_TRIALS
    json_entry["_total_minutes"] = round(total_minutes, 1)
    json_entry["_minutes_per_trial"] = round(total_minutes / N_TRIALS, 1)
    json_entry["embedding"] = args.embedding
    json_entry["aggregation"] = args.aggregation
    json_entry["horizon"] = args.horizon
    json_entry["seed"] = args.seed

    json_path = os.path.join(out_dir, f"optuna_best_params_{args.target}.json")
    existing_json = []
    if os.path.exists(json_path) and os.path.getsize(json_path) > 0:
        with open(json_path, "r") as f:
            existing_json = json.load(f)
    with open(json_path, "w") as f:
        json.dump(existing_json + [json_entry], f, indent=4)
    print(f"Best trial JSON appended to: {json_path}")

    # --- Optuna Visualizations (saved as interactive HTML) ---
    # Single-objective: no target callable needed.
    plots_dir = os.path.join(out_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    optuna.visualization.plot_optimization_history(study).write_html(
        os.path.join(plots_dir, f"optimization_history_{args.target}.html")
    )
    optuna.visualization.plot_parallel_coordinate(study).write_html(
        os.path.join(plots_dir, f"parallel_coordinate_{args.target}.html")
    )
    optuna.visualization.plot_slice(study).write_html(
        os.path.join(plots_dir, f"slice_{args.target}.html")
    )
    optuna.visualization.plot_param_importances(study).write_html(
        os.path.join(plots_dir, f"param_importances_{args.target}.html")
    )
    print(f"Optuna visualizations saved to: {plots_dir}")