import warnings
from pathlib import Path

import holidays
import numpy as np
import pandas as pd
import lightning.pytorch as pl


from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.data import EncoderNormalizer # switched from group normalizer
from pytorch_forecasting.metrics import SMAPE
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor

warnings.filterwarnings("ignore")

SEED = 42
pl.seed_everything(SEED, workers=True)

MACRO_VARS  = ["CPI", "PAYEMS", "INDPRO", "UNRATE", "GDP", "GBP", "YEN", "FFR"]
LAG_VARS    = ["CPI", "PAYEMS", "INDPRO", "UNRATE", "GDP"]
LAG_PERIODS = [1, 2, 6, 12]
# same set as benchmark_runner.LOG_DIFF_TARGETS — log only, no differencing for now
LOG_TARGETS = {"CPI", "PAYEMS", "INDPRO", "GDP"}


class TFTRunner:
    # hyperparams matching initial tryout at  tft-macro.ipynb
    # UPDATED: Change from days to months
    MAX_ENCODER_LENGTH = 24 # 2-year monthly lookback
    MAX_PREDICTION_LENGTH = 12 # 12-month forecast horizon
    PATIENCE = 5
    MAX_EPOCHS = 50
    LEARNING_RATE = 0.03

    # search ranges for Optuna tuning
    TUNE_HIDDEN_SIZE_RANGE         = (16, 128)
    TUNE_HIDDEN_CONTINUOUS_RANGE   = (8, 64)
    TUNE_ATTENTION_HEAD_RANGE      = (1, 4)
    TUNE_DROPOUT_RANGE             = (0.1, 0.4)
    TUNE_LEARNING_RATE_RANGE       = (1e-4, 0.1)
    TUNE_GRADIENT_CLIP_RANGE       = (0.01, 1.0)

    def __init__(self, dfb):
        """
        Parameters
        ----------
        dfb : DataFrameBuilder
            If dfb.load_speech_embeddings() was called before process_data(),
            dfb.pca_cols will be non-empty and those columns will automatically
            be included as time-varying unknown reals.  Otherwise the runner
            operates in macro-only mode.
        """
        self.dfb = dfb
        self._best_params: dict = {}  # populated by tune_hyperparams()

    def tune_hyperparams(
        self,
        train_dl,
        val_dl,
        n_trials: int = 20,
        device: str = 'cpu',
    ) -> dict:
        """Run Optuna via pytorch-forecasting's built-in optimizer.

        Searches over: learning_rate, hidden_size, hidden_continuous_size,
        attention_head_size, dropout, gradient_clip_val.
        Results are stored in self._best_params and returned.

        Parameters
        ----------
        n_trials : int
            Number of Optuna trials (each is a full training run with early stopping).
            20–30 is reasonable; 50+ for a more thorough search.
        """
        from pytorch_forecasting.models.temporal_fusion_transformer.tuning import (
            optimize_hyperparameters,
        )

        import optuna
        accelerator = device if device in ('cuda', 'mps') else 'cpu'
        _study = optuna.create_study(
            direction="minimize",
            sampler=optuna.samplers.TPESampler(seed=SEED),
        )
        study = optimize_hyperparameters(
            train_dl,
            val_dl,
            model_path="optuna_tft",
            n_trials=n_trials,
            study=_study,
            max_epochs=self.MAX_EPOCHS,
            gradient_clip_val_range=self.TUNE_GRADIENT_CLIP_RANGE,
            hidden_size_range=self.TUNE_HIDDEN_SIZE_RANGE,
            hidden_continuous_size_range=self.TUNE_HIDDEN_CONTINUOUS_RANGE,
            attention_head_size_range=self.TUNE_ATTENTION_HEAD_RANGE,
            dropout_range=self.TUNE_DROPOUT_RANGE,
            learning_rate_range=self.TUNE_LEARNING_RATE_RANGE,
            use_learning_rate_finder=False,  # range above handles LR search
            trainer_kwargs=dict(
                accelerator=accelerator,
                devices=1,
                limit_train_batches=30,  # speed up each trial
            ),
        )
        self._best_params = study.best_params
        print(f"[Optuna] best trial #{study.best_trial.number}: {self._best_params}")
        return self._best_params

    def _add_tft_vars(self, df: pd.DataFrame, target: str) -> pd.DataFrame:
        """Add series_id, time_idx, calendar features, monthly lags, and target metadata."""
        df = df.copy()

        df['series_id'] = 'macro'
        df = df.merge(
            df[['date']].drop_duplicates(ignore_index=True).rename_axis('time_idx').reset_index(),
            on='date',
        )

        # log-transform skewed macro vars before lag computation so lags are also in log space
        log_cols = [c for c in LOG_TARGETS if c in df.columns]
        df[log_cols] = np.log(df[log_cols])

        # calendar features — known in advance
        us_holidays = holidays.US()
        df['day_of_week']  = df['date'].dt.dayofweek
        df['week_of_year'] = df['date'].dt.isocalendar().week.astype(int)
        df['month']        = df['date'].dt.month
        df['is_holiday']   = df['date'].dt.date.apply(lambda d: int(d in us_holidays))

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

    def create_tft_dataset(self, train_df, target: str, fold, batch_size: int = 128):
        """
            Build TimeSeriesDataSet and dataloaders. 
            Target is one of MACRO_VARS.
            PCA columns are picked from train_df and appended 
            to time_varying_unknown_reals
        """
        covariates = [v for v in MACRO_VARS if v in train_df.columns]
        lag_cols   = [f"{c}_lag_{l}" for c in LAG_VARS for l in LAG_PERIODS
                      if f"{c}_lag_{l}" in train_df.columns]
        
        # if pca present
        pca_cols_present = [c for c in self.dfb.pca_cols if c in train_df.columns]


        meta_cat_cols  = [f"{v}_{c}" for v in self.dfb.TARGET_COLS
                          for c in ('meta_units', 'meta_frequency')
                          if f"{v}_{c}" in train_df.columns]
        meta_real_cols = [f"{v}_{c}" for v in self.dfb.TARGET_COLS
                          for c in ('meta_popularity', 'meta_years_of_history')
                          if f"{v}_{c}" in train_df.columns]
        
         # hold out the last MAX_PREDICTION_LENGTH rows as a true validation window                                                                                                                                                                    
         # so val_loss reflects out-of-sample fit within the training period                                                                                                                                                                           
        fit_df = train_df.iloc[:-self.MAX_PREDICTION_LENGTH]           

        training = TimeSeriesDataSet(
            data=fit_df,
            # data=train_df,
            time_idx='time_idx',
            target=target,
            group_ids=['series_id'],
            max_encoder_length=self.MAX_ENCODER_LENGTH,
            max_prediction_length=self.MAX_PREDICTION_LENGTH,
            static_categoricals=['series_id'] + meta_cat_cols,
            static_reals=meta_real_cols,
            time_varying_known_reals=['time_idx', 'day_of_week', 'week_of_year', 'month', 'is_holiday',
                                      # also add the fomc dates here since they are well known in advance
                                      # there are like very few exceptions, I think 9/11 was one of them
                                      'days_to_fomc', 'days_since_fomc', 'fomc_cycle_pos', 'meeting_this_month'],
            # speeches are time varying, so i add them here!
            time_varying_unknown_reals= [*covariates, *lag_cols, *pca_cols_present,
                                         # add dissents
                                         'dissent_rate_mean', 'dissent_net_hawk_mean', 'dissent_net_hawk_sum',    
                                         'n_tighter_sum', 'n_easier_sum', 'any_dissent_recent'],
            target_normalizer=EncoderNormalizer(transformation="softplus"),
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
                  use_tqdm: bool = True, use_wandb: bool = False, device: str = 'cpu'):
        """Build trainer + TFT, fit, return best checkpoint path."""
        callbacks = [
            # changed this to val_loss -> train-loss will usually increase, even if val_loss stops
            EarlyStopping(monitor='val_loss', min_delta=1e-2,
                          patience=self.PATIENCE, verbose=False, mode='min'),
        ]
        if use_wandb:
            callbacks.append(LearningRateMonitor())
        if use_tqdm:
            from lightning.pytorch.callbacks.progress import TQDMProgressBar
            callbacks.append(TQDMProgressBar(refresh_rate=2))

        logger = False
        if use_wandb:
            from lightning.pytorch.loggers import WandbLogger
            logger = WandbLogger(project='tft', name='tft-holdout')

        accelerator = device if device in ('cuda', 'mps') else 'cpu'
        # use Optuna best params if tune_hyperparams() was called, else fall back to defaults
        p = self._best_params
        trainer = pl.Trainer(
            max_epochs=self.MAX_EPOCHS,
            accelerator=accelerator,
            devices=1,
            gradient_clip_val=p.get('gradient_clip_val', 0.25),
            # limit_train_batches=60,
            callbacks=callbacks,
            enable_model_summary=True,
            logger=logger,
        )
        tft = TemporalFusionTransformer.from_dataset(
            training_ds,
            learning_rate=p.get('learning_rate', self.LEARNING_RATE),
            lstm_layers=4,
            hidden_size=p.get('hidden_size', 64),
            attention_head_size=p.get('attention_head_size', 2),
            dropout=p.get('dropout', 0.2),
            hidden_continuous_size=p.get('hidden_continuous_size', 8),
            output_size=1,
            loss=SMAPE(),
            log_interval=10,
            reduce_on_plateau_patience=4,
        )
        print(f"[TFT] parameters: {tft.size()/1e3:.1f}k")

        trainer.fit(tft, train_dataloaders=train_dl, val_dataloaders=val_dl)
        return trainer.checkpoint_callback.best_model_path

    def run(self, splits, target: str, fold: int = 0, batch_size: int = 128,
            use_tqdm: bool = True, use_wandb: bool = False, device: str = 'cpu',
            tune: bool = False, n_trials: int = 20) -> str:
        """Full pipeline: augment → dataset → train. Returns best checkpoint path.

        Parameters
        ----------
        tune : bool
            If True, run Optuna hyperparameter search before final training.
        n_trials : int
            Number of Optuna trials (ignored when tune=False).
        """
        self._train_raw = self.dfb.get_data(splits, train=True, model='TFT', fold=fold)
        train_df = self._add_tft_vars(self._train_raw, target)

        self._training_ds, train_dl, val_dl = self.create_tft_dataset(train_df, target, fold, batch_size)
        if tune:
            print(f"[Optuna] Starting hyperparameter search ({n_trials} trials)...")
            self.tune_hyperparams(train_dl, val_dl, n_trials=n_trials, device=device)
        return self.train_tft(self._training_ds, train_dl, val_dl, use_tqdm, use_wandb, device)

    def predict(self, checkpoint_path: str, splits, target: str, fold: int = 0,
                batch_size: int = 128, device: str = 'cpu') -> pd.DataFrame:
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

        test_len = len(test_raw)
        step     = self.MAX_PREDICTION_LENGTH
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

            raw_preds = model.predict(pred_dl)          # (1, MAX_PREDICTION_LENGTH)
            if raw_preds.ndim == 3:
                raw_preds = raw_preds.squeeze(1)
            preds_np = raw_preds.detach().cpu().numpy()

            # for a partial last window the target days sit at the tail of the
            # 360-step output (offset = step - window_size)
            pred_offset = step - window_size

            for i in range(window_size):
                row  = test_window.iloc[i]
                pred = float(preds_np[0, pred_offset + i])
                # reverse the logging from before
                if target in LOG_TARGETS:
                    pred = np.exp(pred)
                # accumulate predictions in original scale for next window's context (no pollution)
                accumulated_preds[row["date"]] = pred
                all_rows.append({
                    "date":      row["date"],
                    "actual":    float(row[target]),  # test_raw is original scale
                    "predicted": pred,
                    "target":    target,
                })

        result_df = pd.DataFrame(all_rows)
        freq = "QS" if target in self.dfb.QUARTERLY_TARGETS else "MS"
        result_df = (
            result_df.set_index("date")
                     .resample(freq).first()
                     .dropna(subset=["actual", "predicted"])
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