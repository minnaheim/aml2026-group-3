
# add necessary variables to the dataframe

# put data into TimeSeriesDataFrame, batchsize, data logger, etc.

# fit tft, add wandb logger

"""
TFT runner for the holdout-validation pipeline.
Trains one multi-series TFT on the training split and evaluates on the test split.
This code simply implements the tft-macro.ipynb and adjusts it to use the data_frame_builder
Code at the bottom let's us run it for now
"""

import warnings
import os
import pandas as pd
import numpy as np
import torch
import lightning.pytorch as pl
import holidays
from pathlib import Path
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.data import GroupNormalizer
from pytorch_forecasting.metrics import SMAPE
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor
from lightning.pytorch.callbacks.progress import TQDMProgressBar

warnings.filterwarnings("ignore")

DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

# hyperparams
PATIENCE         = 5
MAX_EPOCHS       = 1
LEARNING_RATE    = 0.03
BATCH_SIZE       = 128
MAX_ENCODER_LEN  = 48 * 30 # approx 4 years lookback
MAX_PRED_LEN     = 12 * 30 # approx 12 months forecast

LAG_VARS    = ["CPI", "PAYEMS", "INDPRO", "UNRATE", "GDP"]
LAG_PERIODS = [1, 2, 6, 12]


class TFTRunner:
    def __init__(self, dfb):
        self.dfb = dfb

    def _add_features(self, df: pd.DataFrame, train_end: pd.Timestamp) -> pd.DataFrame:
        """Add calendar features, lag features, time_idx, and series_id."""
        df = df.copy()

        # calendar
        us_holidays = holidays.US()
        df["day_of_week"]  = df["date"].dt.dayofweek.astype(float)
        df["week_of_year"] = df["date"].dt.isocalendar().week.astype(float)
        df["month"]        = df["date"].dt.month.astype(float)
        df["is_holiday"]   = df["date"].apply(lambda d: float(d in us_holidays))

        # lag features
        lagged_cols = [f"{col}_lag_{lag}" for col in LAG_VARS for lag in LAG_PERIODS]
        for col in LAG_VARS:
            for lag in LAG_PERIODS:
                df[f"{col}_lag_{lag}"] = df[col].shift(lag)

        # FIX: fill NaNs created by shifting
        # importantly, fill in both directions since the first few values will be missing in the lag of course!
        df[lagged_cols] = df[lagged_cols].ffill().bfill()

        # time index (integer, starting from 0)
        df["time_idx"] = (df["date"] - df["date"].min()).dt.days

        # series_id placeholder (overwritten in long-format reshape)
        df["series_id"] = "all"

        return df, lagged_cols

    def _load_metadata(self, train_end: pd.Timestamp) -> pd.DataFrame:
        meta_path = os.path.join(self.dfb.path, "data", "metadata-macro.csv")
        meta_raw  = pd.read_csv(meta_path, index_col=1)
        meta_raw  = meta_raw.rename(index={
            "CPIAUCSL": "CPI", "DEXUSUK": "GBP", "DEXJPUS": "YEN", "DFF": "FFR"
        })
        meta_raw["meta_years_of_history"] = meta_raw["observation_start"].apply(
            lambda s: max(0.0, (train_end - pd.to_datetime(s)).days / 365.25)
        )
        return meta_raw[["popularity", "units_short", "meta_years_of_history", "frequency_short"]].rename(
            columns={"popularity": "meta_popularity", "units_short": "meta_units",
                     "frequency_short": "meta_frequency"}
        ).loc[LAG_VARS]

    def _to_long(self, df: pd.DataFrame, meta: pd.DataFrame) -> pd.DataFrame:
        """Reshape wide daily df to long format with one row per (date, series)."""
        rows = []
        for var in LAG_VARS:
            subset = df.drop(columns=["series_id"]).copy()
            subset["series_id"] = var
            subset["value"]     = subset[var]
            rows.append(subset)
        df_long = pd.concat(rows, ignore_index=True).sort_values(["series_id", "time_idx"]).reset_index(drop=True)
        for col in meta.columns:
            df_long[col] = df_long["series_id"].map(meta[col])
        return df_long

    def run(self, splits, fold: int = 0) -> pd.DataFrame:
        train_raw = self.dfb.get_data(splits, train=True,  model="TFT", fold=fold)
        test_raw  = self.dfb.get_data(splits, train=False, model="TFT", fold=fold)

        train_end = train_raw["date"].max()
        meta      = self._load_metadata(train_end)

        # add features to full df so lags are consistent across train/test boundary
        full_raw         = pd.concat([train_raw, test_raw], ignore_index=True)
        full_feat, lagged_cols = self._add_features(full_raw, train_end)

        train_feat = full_feat[full_feat["date"] <= train_end].reset_index(drop=True)
        test_feat  = full_feat[full_feat["date"] >  train_end].reset_index(drop=True)

        train_long = self._to_long(train_feat, meta)
        test_long  = self._to_long(test_feat,  meta)

        print(f"[TFT] train: {len(train_long)} rows | test: {len(test_long)} rows | device: {DEVICE}")

        # build datasets
        ds_train = TimeSeriesDataSet(
            data=train_long,
            time_idx="time_idx",
            target="value",
            group_ids=["series_id"],
            max_encoder_length=MAX_ENCODER_LEN,
            max_prediction_length=MAX_PRED_LEN,
            static_categoricals=["series_id", "meta_units"],
            static_reals=["meta_popularity", "meta_years_of_history"],
            time_varying_known_reals=["time_idx", "day_of_week", "week_of_year", "month", "is_holiday"],
            time_varying_known_categoricals=[],
            time_varying_unknown_reals=lagged_cols,
            target_normalizer=GroupNormalizer(groups=["series_id"], transformation="softplus"),
            add_relative_time_idx=True,
            add_target_scales=True,
            add_encoder_length=True,
            allow_missing_timesteps=True,
        )

        ds_val = TimeSeriesDataSet.from_dataset(ds_train, train_long, predict=True, stop_randomization=True)

        train_dl = ds_train.to_dataloader(train=True,  batch_size=BATCH_SIZE,      num_workers=0)
        val_dl   = ds_val.to_dataloader(  train=False, batch_size=BATCH_SIZE * 10, num_workers=0)

        # callbacks
        early_stop = EarlyStopping(monitor="train_loss", min_delta=1e-2, patience=PATIENCE, mode="min")
        lr_logger  = LearningRateMonitor()

        trainer = pl.Trainer(
            max_epochs=MAX_EPOCHS,
            accelerator="auto",
            devices=1,
            gradient_clip_val=0.25,
            limit_train_batches=5,
            callbacks=[lr_logger, early_stop, TQDMProgressBar(refresh_rate=1)],
            enable_model_summary=False,
            logger=True,  # set to WandbLogger if you want tracking
        )

        model = TemporalFusionTransformer.from_dataset(
            ds_train,
            learning_rate=LEARNING_RATE,
            lstm_layers=2,
            hidden_size=16,
            attention_head_size=2,
            dropout=0.2,
            hidden_continuous_size=8,
            output_size=1,
            loss=SMAPE(),
            log_interval=10,
            reduce_on_plateau_patience=4,
        )
        model.to(DEVICE)

        trainer.fit(model, train_dataloaders=train_dl, val_dataloaders=val_dl)

        # predict on validation set (end of training data — same as notebook)
        preds_out = model.predict(val_dl, mode="raw", return_x=True)

        # extract per-series predictions
        results = []
        idx = ds_val.get_parameters()["group_ids"]  # series order
        for var in LAG_VARS:
            sample_idxs = [
                i for i, s in enumerate(preds_out.x["groups"][:, 0].tolist())
                if ds_val.transform_values("series_id", [var], inverse=True) == [s]
                # fallback: match by series_id encoding
            ]
            # simpler approach: filter test_long by series
            actual    = test_long[test_long["series_id"] == var]["value"].values
            # predicted mean from quantile output — shape [n_samples, pred_len]
            predicted = preds_out.output.prediction[..., 0].mean(dim=0).cpu().numpy()[:len(actual)]
            dates     = test_long[test_long["series_id"] == var]["date"].values[:len(predicted)]

            results.append(pd.DataFrame({
                "date":      dates,
                "actual":    actual[:len(predicted)],
                "predicted": predicted,
                "target":    var,
            }))

        return pd.concat(results, ignore_index=True)


# ---------------------------------------------------------------------------
# try it out here!
# ---------------------------------------------------------------------------

from data_frame_builder import DataFrameBuilder

path = r"C:\Users\annaz\OneDrive\Dokumente\Studium\UZH_Master\2026FS\Advanced Machine Learning\Practical_Assignment\aml2026-group-3"
dfb  = DataFrameBuilder(path)
df   = dfb.process_data()
splits, holdout = dfb.generate_split(df)

runner = TFTRunner(dfb)
result = runner.run(splits, fold=0)
print(result)
