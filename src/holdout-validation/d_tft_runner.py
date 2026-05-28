import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import lightning.pytorch as pl

import matplotlib as plt
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.data import EncoderNormalizer # switched from group normalizer
from pytorch_forecasting.metrics import QuantileLoss
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor

warnings.filterwarnings("ignore")

MACRO_VARS  = ["CPI", "PAYEMS", "INDPRO", "UNRATE", "AWHMAN", "USACLI", "GDP",
               # daily vars: mean only — intra-month std has weak theoretical motivation
               "GBP_mean", "YEN_mean", "FFR_mean", "T10Y2Y_mean"]
LAG_VARS    = ["CPI", "PAYEMS", "INDPRO", "UNRATE", "GDP"]
LAG_PERIODS = [1, 12]
# same set as benchmark_runner.LOG_DIFF_TARGETS — log only, no differencing for now
#ANNA# LOG_TARGETS = {"CPI", "PAYEMS", "INDPRO", "GDP"}


HPARAMS_DEFAULTS = {
    "max_encoder_length":     24,
    "max_prediction_length":  12,
    "min_encoder_length":     8,
    "patience":               15,
    "max_epochs":             50,
    "learning_rate":          0.03,
    "lstm_layers":            2,
    "hidden_size":            6,
    "attention_head_size":    1,
    "hidden_continuous_size": 8,
    "dropout":                0.2,
    "normalizer":             "encoder_none",  # "encoder_none" | "group"
}

class TFTRunner:
    QUANTILES = [0.05, 0.1, 0.5, 0.9, 0.95] # forecast quantiles for TFT
    def __init__(self, dfb, hparams: dict | None = None):
        """
        Parameters
        ----------
        dfb : DataFrameBuilder
            If dfb.load_speech_embeddings() was called before process_data(),
            dfb.pca_cols will be non-empty and those columns will automatically
            be included as time-varying unknown reals.  Otherwise the runner
            operates in macro-only mode.
        hparams : dict, optional
            Override any key from HPARAMS_DEFAULTS.
        """
        self.dfb = dfb
        self.hparams = {**HPARAMS_DEFAULTS, **(hparams or {})}
        # convenience aliases (read-only shortcuts used in predict())
        self.MAX_ENCODER_LENGTH    = self.hparams["max_encoder_length"]
        self.MAX_PREDICTION_LENGTH = self.hparams["max_prediction_length"]

    def _add_tft_vars(self, df: pd.DataFrame, target: str) -> pd.DataFrame:
        """Add series_id, time_idx, calendar features, monthly lags, and target metadata."""
        df = df.copy()

        df['series_id'] = 'macro'
        df = df.merge(
            df[['date']].drop_duplicates(ignore_index=True).rename_axis('time_idx').reset_index(),
            on='date',
        )

        #ANNA # log-transform skewed macro vars before lag computation so lags are also in log space
        # log_cols = [c for c in LOG_TARGETS if c in df.columns]
        # df[log_cols] = np.log(df[log_cols])

        # calendar features — known in advance
        df['month'] = df['date'].dt.month  # only meaningful calendar feature on monthly data

        # monthly lags via merge_asof so the offset is in months, not rows
        # (row-based lag is meaningless on ffilled daily data)
        present_lag_vars = [c for c in LAG_VARS if c in df.columns]
        source = df[['date'] + present_lag_vars].sort_values('date').reset_index(drop=True)
        for col in present_lag_vars:
            # GDP is quarterly: lag unit = 1 quarter (3 months); monthly vars: lag unit = 1 month
            months_per_unit = 3 if col in self.dfb.QUARTERLY_TARGETS else 1
            for lag in LAG_PERIODS:
                lookup = pd.DataFrame({
                    'date':  df['date'] - pd.DateOffset(months=lag * months_per_unit),
                    '_orig': range(len(df)),
                }).sort_values('date').reset_index(drop=True)
                merged = pd.merge_asof(lookup, source[['date', col]], on='date', direction='backward')
                df[f"{col}_lag_{lag}"] = merged.sort_values('_orig')[col].values

        # drop rows where 12-month lags are still undefined (first ~12 months)
        df = df.dropna().reset_index(drop=True)

        # static metadata for the target series
        meta_path = Path(self.dfb.path) / 'data' / 'metadata-macro.csv'
        meta_raw  = pd.read_csv(meta_path, index_col=1)
        meta_raw  = meta_raw.rename(index={
            'CPIAUCSL': 'CPI', 'DEXUSUK': 'GBP', 'DEXJPUS': 'YEN', 'DFF': 'FFR'
        })
        train_end = df['date'].max()
        meta_raw['meta_years_of_history'] = meta_raw['observation_start'].apply(
            lambda s: max(0.0, (train_end - pd.to_datetime(s)).days / 365.25)
        )
        meta = meta_raw[['popularity', 'units_short', 'meta_years_of_history', 'frequency_short']].rename(
            columns={'popularity': 'meta_popularity', 'units_short': 'meta_units',
                     'frequency_short': 'meta_frequency'}
        )
        for var in self.dfb.TARGET_COLS:
            if var in meta.index:
                for col, val in meta.loc[var].items():
                    df[f"{var}_{col}"] = val

        return df

    def create_tft_dataset(self, train_df, target: str, fold, batch_size: int = 16):
        """
            Build TimeSeriesDataSet and dataloaders. 
            Target is one of MACRO_VARS.
            PCA columns are picked from train_df and appended 
            to time_varying_unknown_reals
        """
        covariates = [v for v in MACRO_VARS if v in train_df.columns]
        lag_cols   = [f"{c}_lag_{l}" for c in LAG_VARS for l in LAG_PERIODS
                      if f"{c}_lag_{l}" in train_df.columns]
        
        # if pca present (only when --embedding was passed)
        pca_cols_present = [c for c in self.dfb.pca_cols if c in train_df.columns]

        # fomc timing / dissent cols — only present when speeches are loaded
        FOMC_KNOWN   = ['days_to_fomc', 'days_since_fomc', 'fomc_cycle_pos', 'meeting_this_month']
        DISSENT_COLS = ['dissent_rate_mean', 'dissent_net_hawk_mean', 'dissent_net_hawk_sum',
                        'n_tighter_sum', 'n_easier_sum', 'any_dissent_recent']
        fomc_known_present   = [c for c in FOMC_KNOWN   if c in train_df.columns]
        dissent_cols_present = [c for c in DISSENT_COLS if c in train_df.columns]

        meta_cat_cols  = [f"{v}_{c}" for v in self.dfb.TARGET_COLS
                          for c in ('meta_units', 'meta_frequency')
                          if f"{v}_{c}" in train_df.columns]
        meta_real_cols = [f"{v}_{c}" for v in self.dfb.TARGET_COLS
                          for c in ('meta_popularity', 'meta_years_of_history')
                          if f"{v}_{c}" in train_df.columns]
        
         # hold out the last MAX_PREDICTION_LENGTH rows as a true validation window                                                                                                                                                                    
         # so val_loss reflects out-of-sample fit within the training period                                                                                                                                                                           
        fit_df = train_df.iloc[:-self.MAX_PREDICTION_LENGTH]           

        # select normalizer based on hparam
        if self.hparams["normalizer"] == "group":
            from pytorch_forecasting.data import GroupNormalizer
            normalizer = GroupNormalizer(groups=["series_id"])
        elif self.hparams["normalizer"] == "encoder_softplus":
            normalizer = EncoderNormalizer(transformation="softplus")
        else:
            normalizer = EncoderNormalizer(transformation=None) 

        training = TimeSeriesDataSet(
            data=fit_df,
            time_idx='time_idx',
            target=target,
            group_ids=['series_id'],
            max_encoder_length=self.hparams["max_encoder_length"],
            min_encoder_length=self.hparams["min_encoder_length"],
            max_prediction_length=self.hparams["max_prediction_length"],
            static_categoricals=['series_id'] + meta_cat_cols,
            static_reals=meta_real_cols,
            time_varying_known_reals=['time_idx', 'month',
                                      # fomc dates known in advance — only included with speeches
                                      *fomc_known_present],
            time_varying_unknown_reals=[*covariates, *lag_cols, *pca_cols_present, *dissent_cols_present],
            target_normalizer=normalizer,
            add_relative_time_idx=True,
            add_target_scales=True,
            add_encoder_length=True,
            allow_missing_timesteps=True,
        )

        validation = TimeSeriesDataSet.from_dataset(training, train_df, predict=True, stop_randomization=True)
        train_dl   = training.to_dataloader(train=True,  batch_size=batch_size,       num_workers=0)
        val_dl     = validation.to_dataloader(train=False, batch_size=batch_size * 10, num_workers=0)

        return training, train_dl, val_dl

    def train_tft(self, training_ds, train_dl, val_dl,
                  use_tqdm: bool = True, use_wandb: bool = False, device: str = 'cpu',
                  run_name: str = 'tft-holdout'):
        """Build trainer + TFT, fit, return best checkpoint path."""
        callbacks = [
            # changed this to val_loss -> train-loss will usually increase, even if val_loss stops
            EarlyStopping(monitor='val_loss',
                          patience=self.hparams["patience"], verbose=False, mode='min'),
        ]
        if use_wandb:
            callbacks.append(LearningRateMonitor())
        if use_tqdm:
            from lightning.pytorch.callbacks.progress import TQDMProgressBar
            callbacks.append(TQDMProgressBar(refresh_rate=2))

        logger = False
        if use_wandb:
            import wandb
            from lightning.pytorch.loggers import WandbLogger
            logger = WandbLogger(project='tft', name=run_name)
            print(f"  W&B run: {logger.experiment.url}")

        accelerator = device if device in ('cuda', 'mps') else 'cpu'
        trainer = pl.Trainer(
            max_epochs=self.hparams["max_epochs"],
            accelerator=accelerator,
            devices=1,
            gradient_clip_val=0.25,
            callbacks=callbacks,
            enable_model_summary=True,
            logger=logger,
        )

        tft = TemporalFusionTransformer.from_dataset(
            training_ds,
            learning_rate=self.hparams["learning_rate"],
            lstm_layers=self.hparams["lstm_layers"],
            hidden_size=self.hparams["hidden_size"],
            attention_head_size=self.hparams["attention_head_size"],
            dropout=self.hparams["dropout"],
            hidden_continuous_size=self.hparams["hidden_continuous_size"],
            output_size=len(self.QUANTILES),
            loss=QuantileLoss(quantiles=self.QUANTILES),
            log_interval=10,
            reduce_on_plateau_patience=4,
        )
        print(f"[TFT] parameters: {tft.size()/1e3:.1f}k")

        trainer.fit(tft, train_dataloaders=train_dl, val_dataloaders=val_dl)
        ckpt = trainer.checkpoint_callback.best_model_path

        # store for interpret_output
        self._last_ckpt = ckpt
        self._last_training_ds = training_ds
        self._last_val_dl = val_dl
        self._last_device = device
        return ckpt

    def interpret_output(self, out_dir: Path | None = None, target: str = "unknown", embedding_label: str = "none") -> dict:
        """Variable importance + attention from the last trained model.

        Prints encoder/decoder/static importance tables.
        If out_dir is given, saves plot_interpretation figures there.
        """
        from pytorch_forecasting import TemporalFusionTransformer as TFT

        map_loc = self._last_device if self._last_device in ("cuda", "mps") else "cpu"
        model = TFT.load_from_checkpoint(self._last_ckpt, map_location=map_loc)
        model.eval()

        predict_result = model.predict(self._last_val_dl, mode="raw", return_x=True)
        # newer pytorch_forecasting returns NamedTuple(output, x, index); .output is the raw dict
        raw_preds = predict_result.output
        interpretation = model.interpret_output(raw_preds, reduction="mean")
  
        # variable name lists in the same order as the variable-selection networks
        ds = self._last_training_ds
        # if None, just empty list, cannot concat None
        name_map = {
            "static_variables":  (ds.static_reals or []) + (ds.static_categoricals or []),
            "encoder_variables": ((ds.time_varying_unknown_reals or [])
                                  + (ds.time_varying_unknown_categoricals or [])
                                  + (ds.time_varying_known_reals or [])
                                  + (ds.time_varying_known_categoricals or [])),
            "decoder_variables": ((ds.time_varying_known_reals or [])
                                  + (ds.time_varying_known_categoricals or [])),
        }
        importance_dfs = {}
        for key, names in name_map.items():
            if key not in interpretation:
                continue
            vals = interpretation[key].cpu().numpy()
            n = min(len(names), len(vals))
            imp = (
                pd.DataFrame({"variable": names[:n], "importance": vals[:n]})
                .sort_values("importance", ascending=False)
            )
            importance_dfs[key] = imp
            # print(f"\n  {key.replace('_', ' ')}:")
            # print(imp.to_string(index=False))
            
        # save to csv
        if out_dir is not None:
            tar_dir = Path(out_dir) / "interpretation" / f"{target}"
            tar_dir.mkdir(parents=True, exist_ok=True)
            
            combined = pd.concat(
                [imp_df.assign(group=key) for key, imp_df in importance_dfs.items()],
                ignore_index=True
            )
            fname = tar_dir / f"var_selection_h{self.MAX_PREDICTION_LENGTH}_{embedding_label}.csv"
            combined.to_csv(fname, index=False)
            print(f"  Saved: {fname}")

        return importance_dfs

    def run(self, splits, target: str, fold: int = 0, batch_size: int = 16,
            use_tqdm: bool = True, use_wandb: bool = False, device: str = 'cpu',
            run_name: str = 'tft-holdout') -> str:
        """Full pipeline: augment → dataset → train. Returns best checkpoint path."""
        self._train_raw = self.dfb.get_data(splits, train=True, model='TFT', fold=fold)
        print("Got data from data_frame_builder...")
        train_df = self._add_tft_vars(self._train_raw, target)
        print("*************************** columns of the train_df (to check whether embeddings inside) ***************************")
        print(train_df.columns[1:50])

        self._training_ds, train_dl, val_dl = self.create_tft_dataset(train_df, target, fold, batch_size)
        print("Created TimeSeriesDataSet for tft...")
        return self.train_tft(self._training_ds, train_dl, val_dl, use_tqdm, use_wandb, device, run_name)

    def predict(self, checkpoint_path: str, splits, target: str, fold: int = 0,
                batch_size: int = 16, device: str = 'cpu', step: int | None = None) -> pd.DataFrame:
        """Rolling-window inference over the test split.

        The test period is typically much longer than MAX_PREDICTION_LENGTH, so
        predictions are made in non-overlapping windows of MAX_PREDICTION_LENGTH days.
        Returns a monthly (or quarterly for GDP) DataFrame with date/actual/predicted/target.
        """
        

        # fetch model params & evaluate
        map_location = device if device in ("cuda", "mps") else "cpu"
        # this is written in train_tft
        model = TemporalFusionTransformer.load_from_checkpoint(checkpoint_path, map_location=map_location)
        model.eval()

        # check if we have the training_ds & train_raw available (from run method) if not, re-create
        train_raw   = getattr(self, '_train_raw',   self.dfb.get_data(splits, train=True,  model='TFT', fold=fold))
        training_ds = getattr(self, '_training_ds', None)
        # TODO: why would it be? run() before predict()...
        # if training_ds is None:
        #     train_df    = self._add_tft_vars(train_raw, target)
        #     training_ds, _, _ = self.create_tft_dataset(train_df, target, fold, batch_size)

        # get validation data (not test. misnomer), test not used in cv
        test_raw = self.dfb.get_data(splits, train=False, model='TFT', fold=fold)

        test_len  = len(test_raw)
        step      = step if step is not None else self.MAX_PREDICTION_LENGTH
        all_rows: list[dict] = []
        accumulated_preds: dict = {}  # date -> prediction in original scale (no pollution)

        for start in range(0, test_len, step):
            end         = min(start + step, test_len)
            window_size = end - start
            test_window = test_raw.iloc[start:end]

            # context = train + test[0:end]; for the last partial window we still
            # include test[0:start+step] (or up to test_len) so predict=True gets
            # a full MAX_PREDICTION_LENGTH decoder window to index into.
            context_end = min(start + step, test_len)
            context_raw = pd.concat([train_raw, test_raw.iloc[:context_end]], ignore_index=True)
            # replace target values in test portion with own predictions (no actual test obs — no pollution)
            if accumulated_preds:
                # mask the values which have been predicted (= accumulated)
                mask = context_raw["date"].isin(accumulated_preds)
                context_raw.loc[mask, target] = context_raw.loc[mask, "date"].map(accumulated_preds)
            context_df  = self._add_tft_vars(context_raw, target)

            pred_ds = TimeSeriesDataSet.from_dataset(
                training_ds, context_df, predict=True, stop_randomization=True
            )
            pred_dl = pred_ds.to_dataloader(train=False, batch_size=batch_size, num_workers=0)

            raw_preds = model.predict(pred_dl, mode="quantiles")  # (1, MAX_PREDICTION_LENGTH)
            if raw_preds.ndim == 3:
                raw_preds = raw_preds.squeeze(1)
            preds_np = raw_preds.detach().cpu().numpy()

            # model always outputs MAX_PREDICTION_LENGTH steps with predict=True;
            # for a partial last window the target days sit at the tail of that output
            pred_offset = self.MAX_PREDICTION_LENGTH - window_size

            for i in range(window_size):
                row  = test_window.iloc[i]
                pred = float(preds_np[0, pred_offset + i, self.QUANTILES.index(0.5)]) # median quantile is used for prediction  
                #ANNA # reverse the logging from before
                # if target in LOG_TARGETS:
                #     pred = np.exp(pred)
                # accumulate predictions in original scale for next window's context (no pollution)
                accumulated_preds[row["date"]] = pred
                all_rows.append({
                    "date":      row["date"],
                    "actual":    float(row[target]),  # test_raw is original scale
                    "predicted": float(preds_np[0, pred_offset + i, self.QUANTILES.index(0.5)]),   # q0.5
                    "pred_lo":   float(preds_np[0, pred_offset + i, self.QUANTILES.index(0.1)]),   # q0.1
                    "pred_hi":   float(preds_np[0, pred_offset + i, self.QUANTILES.index(0.9)]),   # q0.9
                    "target":    target,
                    "step":      i + 1,        # ← add this
                    "window":    start // step,
                })

        result_df = pd.DataFrame(all_rows)
        freq = "QS" if target in self.dfb.QUARTERLY_TARGETS else "MS"
        result_df = (
            result_df.set_index("date")
                     .resample(freq).first()
                     .dropna(subset=["actual", "predicted", "pred_lo", "pred_hi"])  # drop any rows where actual or predicted is NaN
                     .reset_index()
        )
        result_df["target"] = target
        return result_df



# # try this out!
# from data_frame_builder import DataFrameBuilder

# # path = "/Users/minna/Code/FS26/AML/aml2026-group-3"
# path = r"C:\Users\annaz\OneDrive\Dokumente\Studium\UZH_Master\2026FS\Advanced Machine Learning\Practical_Assignment\aml2026-group-3"

# # # without speeches
# # dfb  = DataFrameBuilder(path) 

# # with speeches
# # right now, only fomc roberta is on github; will allow for finbert later on AND kafka speeches + reshuffling
# # then, update embeddings registry in data_frame_builder.py
# dfb  = DataFrameBuilder(path, embedding="fomc-roberta")
# df   = dfb.process_data()

# splits, holdout = dfb.generate_split(df)

# for s in splits:
#    tr, te = s["train"], s["test"]
#    print(f"Fold {s['fold']}: train [{tr['date'].min().date()} – {tr['date'].max().date()}] ({len(tr)} rows) | "
#          f"test [{te['date'].min().date()} – {te['date'].max().date()}] ({len(te)} rows)")

# tftr = TFTRunner(dfb)
# ckpt = tftr.run(splits=splits, target="CPI", device = 'cpu', batch_size = 64)
# print(f"Best checkpoint: {ckpt}")


# # look at results
# preds = tftr.predict(ckpt, splits=splits, target="CPI", batch_size = 64)

# import matplotlib.pyplot as plt

# fig, ax = plt.subplots(figsize=(12, 4))
# ax.plot(preds["date"], preds["actual"],    label="Actual",    color="steelblue")
# ax.plot(preds["date"], preds["predicted"], label="Predicted", color="tomato", linestyle="--")
# ax.set_title("TFT — CPI (test period)")
# ax.set_xlabel("Date")
# ax.set_ylabel("CPI")
# ax.legend()
# plt.tight_layout()
# plt.show()