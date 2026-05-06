# -*- coding: utf-8 -*-
"""
TFT model: quarterly macro prediction
======================================

Target:   GDP (quarterly, level in billions USD)  [default]
          CPI (monthly index, collapsed to quarterly)
Inputs:   - Economic macro variables (last observed value before each quarter start)
            Monthly: CPI, PAYEMS, INDPRO, UNRATE
            Daily:   GBP, YEN, FFR
          - Speech PCA embeddings (fomc-roberta, mean-pooled over rolling 4-quarter
            window of speeches given *before* the quarter start)

Design:   One row per quarter Q.  All covariates represent information available at
          the beginning of Q (i.e. from the previous quarter).  No look-ahead.

Split:    train  : date < CUTOFF_DATE
          test   : date >= CUTOFF_DATE

Usage:
    python tft_gdp_cpi_basic.py [--target {GDP,CPI}] [--no-speeches]

Arguments:
    --target {GDP,CPI}   Variable to predict. Default: GDP.
    --no-speeches        Exclude FOMC speech PCA embeddings;
                         train on macro covariates only.

Examples:
    # Predict GDP with speech embeddings (default)
    python tft_gdp_cpi_basic.py

    # Predict GDP without speech embeddings
    python tft_gdp_cpi_basic.py --no-speeches

    # Predict CPI with speech embeddings
    python tft_gdp_cpi_basic.py --target CPI

    # Predict CPI without speech embeddings
    python tft_gdp_cpi_basic.py --target CPI --no-speeches
"""

import argparse
import os
import sys
import warnings
warnings.filterwarnings("ignore")

_parser = argparse.ArgumentParser(description="TFT quarterly macro prediction")
_parser.add_argument(
    "--target", dest="target", default="GDP", choices=["GDP", "CPI"],
    help="Target variable to predict (default: GDP).",
)
_parser.add_argument(
    "--no-speeches", dest="no_speeches", action="store_true",
    help="Exclude FOMC speech PCA embeddings; train on macro variables only.",
)
_parser.add_argument(
    "--shuffled-speeches", dest="shuffled_speeches", action="store_true",
    help="Use shuffled (randomised) speech embeddings instead of real ones.",
)
_parser.add_argument(
    "--embedding-file", dest="embedding_file", default=None,
    help="Path to a speech embeddings CSV (relative to repo root or absolute). "
         "Overrides the default fomc-roberta file. e.g. "
         "data/embeddings/fomc-roberta/embeddings_pca_cls_full_fomc-roberta.csv",
)
_args = _parser.parse_args()
TARGET       = _args.target
USE_SPEECHES = not _args.no_speeches
USE_SHUFFLED = _args.shuffled_speeches
EMBEDDING_FILE = _args.embedding_file  # None → use defaults

# Ensure stdout handles unicode on Windows terminals
if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd
import torch
torch.set_float32_matmul_precision("medium")   # faster on Tensor Core GPUs
import lightning.pytorch as pl
import matplotlib
matplotlib.use("Agg")          # headless – swap to "TkAgg" / "Qt5Agg" if interactive
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet, Baseline
from pytorch_forecasting.data import GroupNormalizer, EncoderNormalizer
from pytorch_forecasting.metrics import QuantileLoss, SMAPE
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor
from lightning.pytorch.loggers import CSVLogger

# ── Config ─────────────────────────────────────────────────────────────────────
REPO_PATH   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CUTOFF_DATE = pd.Timestamp("2018-01-01")   # train < CUTOFF, test >= CUTOFF

MAX_ENCODER = 20    # quarters of look-back  (5 years)
MAX_PRED    = 1     # one quarter ahead
MIN_ENCODER = 8     # allow shorter context near series start

BATCH_SIZE  = 16
MAX_EPOCHS  = 60
PATIENCE    = 8
LR          = 0.005

SEED = 42
pl.seed_everything(SEED, workers=True)

SPEECH_WINDOW_QUARTERS = 4        # rolling window for PCA aggregation (1 year)
START_DATE = pd.Timestamp("1990-01-01")    # earliest quarter to use

# USE_SPEECHES is set via --no-speeches CLI flag (see argparse block above)

DEVICE = ("mps"  if torch.backends.mps.is_available()  else
          "cuda" if torch.cuda.is_available()           else "cpu")

QUANTILES = [0.1, 0.25, 0.5, 0.75, 0.9]   # TFT output quantiles

# ── 1. Load raw data ────────────────────────────────────────────────────────────
gdp = (pd.read_csv(f"{REPO_PATH}/data/macro-vars-quarterly.csv",
                   index_col=0, parse_dates=True)["GDP"]
         .dropna()
         .sort_index())

monthly = (pd.read_csv(f"{REPO_PATH}/data/macro-vars-monthly.csv",
                       index_col=0, parse_dates=True)
             [["CPI", "PAYEMS", "INDPRO", "UNRATE"]]     # stable, long-history series
             .sort_index())

daily = (pd.read_csv(f"{REPO_PATH}/data/macro-vars-daily.csv",
                     index_col=0, parse_dates=True)
           [["GBP", "YEN", "FFR"]]                        # longest coverage
           .sort_index())

if EMBEDDING_FILE is not None:
    _speech_path = (EMBEDDING_FILE if os.path.isabs(EMBEDDING_FILE)
                    else os.path.join(REPO_PATH, EMBEDDING_FILE))
elif USE_SHUFFLED:
    _speech_path = f"{REPO_PATH}/data/embeddings/fomc-roberta/shuffled/embeddings_pca_cls_full_fomc-roberta_shuffled.csv"
else:
    _speech_path = f"{REPO_PATH}/data/embeddings/fomc-roberta/embeddings_pca_mean_full_fomc-roberta.csv"

# Derive a human-readable label from the filename (e.g. "cls_full", "mean_512")
_emb_stem = os.path.splitext(os.path.basename(_speech_path))[0]  # strip .csv
EMBEDDING_VARIANT = _emb_stem  # full stem as fallback
for _marker in ("embeddings_pca_", "fomc-roberta", "finbert", "llama"):
    _emb_stem = _emb_stem.replace(_marker, "")
EMBEDDING_VARIANT = _emb_stem.strip("_-") or EMBEDDING_VARIANT

speeches = pd.read_csv(_speech_path, parse_dates=["Date"]).sort_values("Date").reset_index(drop=True)

PCA_COLS  = sorted([c for c in speeches.columns if c.startswith("pca_")])
MACRO_COLS = list(monthly.columns) + list(daily.columns)

print(f"GDP:       {gdp.index.min().date()} -> {gdp.index.max().date()}  ({len(gdp)} quarters)")
print(f"Monthly:   {monthly.index.min().date()} -> {monthly.index.max().date()}")
print(f"Daily:     {daily.index.min().date()} -> {daily.index.max().date()}")
print(f"Speeches:  {speeches['Date'].min().date()} -> {speeches['Date'].max().date()}  "
      f"({len(speeches)} speeches, {len(PCA_COLS)} PCA dims)")


# ── 2. Build quarterly feature table ───────────────────────────────────────────

def last_before(series: pd.Series, date: pd.Timestamp) -> float:
    """Last non-null value with index strictly before *date*."""
    sub = series.loc[series.index < date].dropna()
    return float(sub.iloc[-1]) if len(sub) else np.nan


rows = []
for q_date in gdp.index:
    row: dict = {"date": q_date, "GDP": gdp[q_date]}

    # --- macro covariates: last value published before quarter start ----------
    for col in monthly.columns:
        row[col] = last_before(monthly[col], q_date)
    for col in daily.columns:
        row[col] = last_before(daily[col], q_date)

    # --- speech PCAs: mean over speeches in rolling window before q_date ------
    if USE_SPEECHES:
        window_start = q_date - pd.DateOffset(months=3 * SPEECH_WINDOW_QUARTERS)
        mask = (speeches["Date"] >= window_start) & (speeches["Date"] < q_date)
        sub  = speeches.loc[mask, PCA_COLS]
        if len(sub) == 0:
            # fallback: any speech before q_date (handles early quarters)
            sub = speeches.loc[speeches["Date"] < q_date, PCA_COLS]
        for col in PCA_COLS:
            row[col] = float(sub[col].mean()) if len(sub) else np.nan

    rows.append(row)

df = pd.DataFrame(rows)
df = df[df["date"] >= START_DATE].dropna().reset_index(drop=True)
df["time_idx"] = range(len(df))
df["series"]   = TARGET

# Exclude the chosen target from covariates so there is no leakage
COVARIATE_COLS = [c for c in MACRO_COLS + (PCA_COLS if USE_SPEECHES else []) if c != TARGET]

print(f"\nQuarterly dataset: {len(df)} rows  "
      f"({df['date'].min().date()} -> {df['date'].max().date()})")
print(f"Target:            {TARGET}")
speech_status = ("OFF" if not USE_SPEECHES else "ON (shuffled)" if USE_SHUFFLED else "ON")
print(f"Speech embeddings: {speech_status}")
print(f"Covariates ({len(COVARIATE_COLS)}): {COVARIATE_COLS}")


# ── 3. Train / test split ───────────────────────────────────────────────────────
train_df = df[df["date"] <  CUTOFF_DATE].reset_index(drop=True)
test_df  = df[df["date"] >= CUTOFF_DATE].reset_index(drop=True)

print(f"\nTrain: {len(train_df)} quarters "
      f"({train_df['date'].min().date()} -> {train_df['date'].max().date()})")
print(f"Test:  {len(test_df)} quarters "
      f"({test_df['date'].min().date()} -> {test_df['date'].max().date()})")


# ── 4. TimeSeriesDataSet ────────────────────────────────────────────────────────
training_ds = TimeSeriesDataSet(
    data                      = train_df,
    time_idx                  = "time_idx",
    target                    = TARGET,
    group_ids                 = ["series"],
    min_encoder_length        = MIN_ENCODER,
    max_encoder_length        = MAX_ENCODER,
    min_prediction_length     = 1,
    max_prediction_length     = MAX_PRED,
    static_categoricals       = ["series"],
    time_varying_known_reals  = ["time_idx"],
    time_varying_unknown_reals= COVARIATE_COLS,
    # EncoderNormalizer: normalises each sample by its own encoder-window stats,
    # so the model generalises to out-of-sample GDP levels beyond the training range.
    target_normalizer         = EncoderNormalizer(transformation="softplus"),
    add_relative_time_idx     = True,
    add_target_scales         = True,
    add_encoder_length        = True,
    allow_missing_timesteps   = False,
)

# validation: last MAX_PRED quarter(s) of the training period
val_ds = TimeSeriesDataSet.from_dataset(training_ds, train_df,
                                        predict=True, stop_randomization=True)

train_dl = training_ds.to_dataloader(train=True,  batch_size=BATCH_SIZE,    num_workers=0)
val_dl   = val_ds.to_dataloader(     train=False, batch_size=BATCH_SIZE * 4, num_workers=0)


# ── 5. Build TFT ────────────────────────────────────────────────────────────────
tft = TemporalFusionTransformer.from_dataset(
    training_ds,
    learning_rate          = LR,
    lstm_layers            = 2,
    hidden_size            = 64,
    attention_head_size    = 4,
    dropout                = 0.1,
    hidden_continuous_size = 32,
    output_size            = len(QUANTILES),    # one output per quantile
    loss                   = QuantileLoss(quantiles=QUANTILES),
    log_interval           = 10,
    reduce_on_plateau_patience = 4,
)
print(f"\nTFT parameters: {tft.size()/1e3:.1f}k")


# ── 6. Train ────────────────────────────────────────────────────────────────────
trainer = pl.Trainer(
    max_epochs       = MAX_EPOCHS,
    accelerator      = "auto",
    devices          = 1,
    gradient_clip_val= 0.1,
    deterministic    = "warn",
    callbacks        = [
        EarlyStopping(monitor="val_loss", min_delta=1e-4,
                      patience=PATIENCE, mode="min"),
        LearningRateMonitor(),
    ],
    enable_model_summary = True,
    logger               = CSVLogger(save_dir=os.path.join(REPO_PATH, "src", "logs"),
                                     name=f"tft_{TARGET.lower()}_" + ("speeches" if USE_SPEECHES else "macro_only")),


)

print("\nTraining …")
trainer.fit(tft, train_dataloaders=train_dl, val_dataloaders=val_dl)

best_tft = TemporalFusionTransformer.load_from_checkpoint(
    trainer.checkpoint_callback.best_model_path)
print(f"Best checkpoint: {trainer.checkpoint_callback.best_model_path}")


# ── 7. Rolling 1-step-ahead test evaluation ─────────────────────────────────────
# For each test quarter Q we build a mini-window:
#   context = last MAX_ENCODER quarters from the full df ending at Q-1
#   target  = quarter Q
# The time_idx within each window is remapped to 0..n to keep the dataset
# constructor happy (the trained normalizer is applied statefully, so the
# actual integer values do not matter for inference).

print("\nRolling 1-step-ahead predictions on test set …")

test_dates    = []
test_actuals  = []
test_pred_med = []    # median (quantile 0.5)
test_pred_lo  = []    # quantile 0.1
test_pred_hi  = []    # quantile 0.9

MED_IDX = QUANTILES.index(0.5)
LO_IDX  = QUANTILES.index(0.1)
HI_IDX  = QUANTILES.index(0.9)

for _, pred_row in test_df.iterrows():
    q_time_idx = pred_row["time_idx"]

    # grab up to MAX_ENCODER quarters of context from full df (train + test)
    ctx = df[df["time_idx"] < q_time_idx].tail(MAX_ENCODER).copy()
    if len(ctx) < MIN_ENCODER:
        print(f"  Skip {pred_row['date'].date()}: only {len(ctx)} context quarters")
        continue

    window = pd.concat([ctx, pred_row.to_frame().T], ignore_index=True).copy()
    # remap time_idx to consecutive integers starting from 0
    window["time_idx"] = range(len(window))
    window["series"]   = TARGET
    # ensure correct dtypes after concat
    for col in COVARIATE_COLS + [TARGET]:
        window[col] = pd.to_numeric(window[col], errors="coerce")

    try:
        pred_ds = TimeSeriesDataSet.from_dataset(
            training_ds, window, predict=True, stop_randomization=True)
        pred_dl = pred_ds.to_dataloader(
            train=False, batch_size=1, num_workers=0)

        q_preds_tensor = best_tft.predict(pred_dl, mode="quantiles")
        # shape: (batch=1, time=1, n_quantiles)
        q_preds = q_preds_tensor[0, 0, :].cpu().numpy()  # (n_quantiles,)

        test_dates.append(pred_row["date"])
        test_actuals.append(float(pred_row[TARGET]))
        test_pred_med.append(float(q_preds[MED_IDX]))
        test_pred_lo.append(float(q_preds[LO_IDX]))
        test_pred_hi.append(float(q_preds[HI_IDX]))

    except Exception as exc:
        print(f"  Skip {pred_row['date'].date()}: {exc}")


# ── 8. Metrics ──────────────────────────────────────────────────────────────────
test_dates    = pd.to_datetime(test_dates)
test_actuals  = np.array(test_actuals)
test_pred_med = np.array(test_pred_med)

mae  = np.mean(np.abs(test_actuals - test_pred_med))
smape_val = np.mean(2 * np.abs(test_actuals - test_pred_med) /
                    (np.abs(test_actuals) + np.abs(test_pred_med) + 1e-8)) * 100

print(f"\n── Test-set results ({len(test_dates)} quarters) [{TARGET}] ──────────────────")
print(f"  MAE   = {mae:,.4g}")
print(f"  SMAPE = {smape_val:.2f}%")

# ── Output directory ─────────────────────────────────────────────────────────────
if not USE_SPEECHES:
    suffix = "macro_only"
elif USE_SHUFFLED:
    suffix = f"shuffled_{EMBEDDING_VARIANT}"
else:
    suffix = f"speeches_{EMBEDDING_VARIANT}"
OUT_DIR  = os.path.join(REPO_PATH, "out", "tft")
os.makedirs(OUT_DIR, exist_ok=True)

# ── 8b. Append metrics to shared .txt ───────────────────────────────────────────
if not USE_SPEECHES:
    embedding_label = "OFF"
elif USE_SHUFFLED:
    embedding_label = f"ON – shuffled ({EMBEDDING_VARIANT})"
else:
    embedding_label = f"ON – {EMBEDDING_VARIANT}"

metrics_path = os.path.join(OUT_DIR, "metrics_all_runs.txt")
with open(metrics_path, "a") as f:
    f.write("\n" + "=" * 55 + "\n")
    f.write(f"Run date:          {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}\n")
    f.write(f"Target:            {TARGET}\n")
    f.write(f"Speech embeddings: {embedding_label}\n")
    f.write(f"Seed:              {SEED}\n")
    f.write(f"Cutoff date:       {CUTOFF_DATE.date()}\n")
    f.write(f"Test quarters:     {len(test_dates)}\n")
    f.write(f"  ({test_dates.min().date()} -> {test_dates.max().date()})\n")
    f.write(f"\nMAE   = {mae:,.4g}\n")
    f.write(f"SMAPE = {smape_val:.2f}%\n")
print(f"Metrics appended -> {metrics_path}")

# ── 9. Plot ─────────────────────────────────────────────────────────────────────
_units = {"GDP": "Billions USD", "CPI": "Index (1982-84=100)"}
y_label = f"{TARGET} ({_units.get(TARGET, TARGET)})"

fig, ax = plt.subplots(figsize=(13, 5))

# Full target series for context
ax.plot(df["date"], df[TARGET], color="steelblue", linewidth=1.4,
        label=f"Actual {TARGET} (full series)")

# Train / test divider
ax.axvline(CUTOFF_DATE, color="grey", linestyle=":", linewidth=1.2)
ax.text(CUTOFF_DATE, ax.get_ylim()[0], "  train | test",
        va="bottom", ha="left", fontsize=8, color="grey")

# Test predictions
ax.plot(test_dates, test_pred_med, color="tomato", linewidth=1.8,
        linestyle="--", label=f"TFT median (SMAPE={smape_val:.1f}%)")
ax.fill_between(test_dates,
                test_pred_lo, test_pred_hi,
                color="tomato", alpha=0.15,
                label="10\u201390% prediction interval")

ax.set_title(f"TFT: 1-quarter-ahead {TARGET} prediction  |  quantile output  |  embeddings: {embedding_label}")
ax.set_xlabel("Date")
ax.set_ylabel(y_label)
ax.legend(loc="upper left")
plt.tight_layout()

plot_path = os.path.join(OUT_DIR, f"tft_{TARGET.lower()}_predictions_{suffix}.png")
plt.savefig(plot_path, dpi=150)
print(f"Plot saved    -> {plot_path}")
plt.close()


# ── 10. Variable importance (TFT interpretability) ───────────────────────────────
# TFT's variable selection networks assign a weight to each input at every
# time step.  We accumulate those weights over the full training set to get
# stable importance scores.
print("\nComputing variable importance …")

raw_preds = best_tft.predict(
    train_dl, mode="raw", trainer_kwargs={"accelerator": "auto"}
)
interpretation = best_tft.interpret_output(raw_preds, reduction="sum")

# ── 10a. Plot (using built-in helper) ────────────────────────────────────────
figs = best_tft.plot_interpretation(interpretation)
for fig_name, fig in figs.items():
    fig.suptitle(f"Target: {TARGET}  |  embeddings: {embedding_label}", fontsize=9, y=1.01)
    fig_path = os.path.join(OUT_DIR, f"importance_{TARGET.lower()}_{suffix}_{fig_name}.png")
    fig.savefig(fig_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"Importance plot saved -> {fig_path}")

# ── 10b. Save ranked importance table as CSV ─────────────────────────────────
importance_rows = []
label_map = {
    "encoder_variables": "encoder (past covariates)",
    "decoder_variables": "decoder (future-known)",
    "static_variables":  "static",
}
for key, label in label_map.items():
    if key not in interpretation:
        continue
    scores = interpretation[key].cpu().numpy()
    names  = (
        best_tft.encoder_variables if key == "encoder_variables"
        else best_tft.decoder_variables if key == "decoder_variables"
        else best_tft.static_variables
    )
    for name, score in sorted(zip(names, scores), key=lambda t: -t[1]):
        importance_rows.append({"role": label, "variable": name, "importance": round(float(score), 6)})

importance_df = pd.DataFrame(importance_rows)
importance_csv = os.path.join(OUT_DIR, f"importance_{TARGET.lower()}_{suffix}.csv")
importance_df.to_csv(importance_csv, index=False)
print(f"Importance table saved -> {importance_csv}")
print("\n── Variable importance (encoder) ─────────────────────────────────────────")
print(importance_df[importance_df["role"] == "encoder (past covariates)"]
      .head(20).to_string(index=False))
