import warnings
import pandas as pd
import lightning.pytorch as pl

from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.data import GroupNormalizer
from pytorch_forecasting.metrics import SMAPE
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor

warnings.filterwarnings("ignore")

MACRO_VARS = ["CPI", "PAYEMS", "INDPRO", "UNRATE", "GDP", "GBP", "YEN", "FFR"]


class TFTRunner:
    # hyperparams matching initial tryout at  tft-macro.ipynb
    MAX_ENCODER_LENGTH    = 48 * 30   # 4-year lookback
    MAX_PREDICTION_LENGTH = 12 * 30   # 12-month forecast horizon
    PATIENCE      = 5
    MAX_EPOCHS    = 40
    LEARNING_RATE = 0.03

    def __init__(self, dfb):
        self.dfb = dfb

    def _add_tft_vars(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add series_id and time_idx — minimum required by TimeSeriesDataSet."""
        df = df.copy()
        # only these 2 are necessary to construct TimeSeriesDataSet
        df['series_id'] = 'macro'
        df = df.merge(
            df[['date']].drop_duplicates(ignore_index=True).rename_axis('time_idx').reset_index(),
            on='date',
        )
        return df

    # def _fetch_data(self, splits, target: str = "CPI", fold: int = 0):
    #     train = self.dfb.get_data(splits, train=True,  model="TFT", target=target, fold=fold)
    #     test  = self.dfb.get_data(splits, train=False, model="TFT", target=target, fold=fold)
    #     return train, test

    def create_tft_dataset(self, train_df, target: str,fold,  batch_size: int = 128):
        """Build TimeSeriesDataSet and dataloaders. target is one of MACRO_VARS."""

        # fetch train_df from dfb
        # train_df, _ = self._fetch_data(splits, target, fold)
        # print(train_df.columns)

        # create TimeSeriesDataSet (uniquely created for TFT)
        covariates = [v for v in MACRO_VARS if v in train_df.columns] # target included if it is an unkown real -> ours are!

        training = TimeSeriesDataSet(
            data=train_df,
            time_idx='time_idx',
            target=target,
            group_ids=['series_id'],
            max_encoder_length=self.MAX_ENCODER_LENGTH,
            max_prediction_length=self.MAX_PREDICTION_LENGTH,
            static_categoricals=['series_id'],
            time_varying_known_reals=['time_idx'],
            time_varying_unknown_reals=covariates,
            target_normalizer=GroupNormalizer(groups=['series_id'], transformation='softplus'),
            add_relative_time_idx=True,
            add_target_scales=True,
            add_encoder_length=True,
            allow_missing_timesteps=True,
        )

        # TODO: test this well!!!
        validation = TimeSeriesDataSet.from_dataset(training, train_df, predict=True, stop_randomization=True)
        train_dl   = training.to_dataloader(train=True,  batch_size=batch_size,       num_workers=0)
        val_dl     = validation.to_dataloader(train=False, batch_size=batch_size * 10, num_workers=0)

        return training, train_dl, val_dl

    def train_tft(self, training_ds, train_dl, val_dl,
                  use_tqdm: bool = True, use_wandb: bool = False, device: str = 'cpu'):
        """Build trainer + TFT, fit, return best checkpoint path."""
        callbacks = [
            EarlyStopping(monitor='train_loss', min_delta=1e-2,
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
        trainer = pl.Trainer(
            max_epochs=self.MAX_EPOCHS,
            accelerator=accelerator,
            devices=1,
            gradient_clip_val=0.25,
            limit_train_batches=5,
            callbacks=callbacks,
            enable_model_summary=True,
            logger=logger,
        )

        tft = TemporalFusionTransformer.from_dataset(
            training_ds,
            learning_rate=self.LEARNING_RATE,
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
        print(f"[TFT] parameters: {tft.size()/1e3:.1f}k")

        trainer.fit(tft, train_dataloaders=train_dl, val_dataloaders=val_dl)
        return trainer.checkpoint_callback.best_model_path

    def run(self, splits, target: str, fold: int = 0, batch_size: int = 128,
            use_tqdm: bool = True, use_wandb: bool = False, device: str = 'cpu') -> str:
        """Full pipeline: augment → dataset → train. Returns best checkpoint path."""
        train_raw = self.dfb.get_data(splits, train=True, model='TFT', fold=fold)
        print("Got data from data_frame_builder...")
        train_df  = self._add_tft_vars(train_raw)
        training_ds, train_dl, val_dl = self.create_tft_dataset(train_df, target, fold, batch_size)
        print("Created TimeSeriesDataSet for tft...")
        return self.train_tft(training_ds, train_dl, val_dl, use_tqdm, use_wandb, device)

    def predict(self, checkpoint_path: str, splits, target: str, fold: int = 0,
                batch_size: int = 128, device: str = 'cpu') -> pd.DataFrame:
        """Rolling-window inference over the test split.

        The test period is typically much longer than MAX_PREDICTION_LENGTH, so
        predictions are made in non-overlapping windows of MAX_PREDICTION_LENGTH days.
        Returns a monthly (or quarterly for GDP) DataFrame with date/actual/predicted/target.
        """
        # local import to avoid circular import at module level
        from pytorch_forecasting import TemporalFusionTransformer as TFT

        map_location = device if device in ("cuda", "mps") else "cpu"
        model = TFT.load_from_checkpoint(checkpoint_path, map_location=map_location)
        model.eval()

        train_raw = self.dfb.get_data(splits, train=True,  model='TFT', fold=fold)
        test_raw  = self.dfb.get_data(splits, train=False, model='TFT', fold=fold)

        # build reference training dataset (data manipulation only — no retraining)
        train_df    = self._add_tft_vars(train_raw)
        training_ds, _, _ = self.create_tft_dataset(train_df, target, fold, batch_size)

        test_len = len(test_raw)
        step     = self.MAX_PREDICTION_LENGTH
        all_rows: list[dict] = []

        for start in range(0, test_len, step):
            end         = min(start + step, test_len)
            window_size = end - start
            test_window = test_raw.iloc[start:end]

            # context = train + test[0:end]; for the last partial window we still
            # include test[0:start+step] (or up to test_len) so predict=True gets
            # a full MAX_PREDICTION_LENGTH decoder window to index into.
            context_end = min(start + step, test_len)
            context_raw = pd.concat([train_raw, test_raw.iloc[:context_end]], ignore_index=True)
            context_df  = self._add_tft_vars(context_raw)

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
                row = test_window.iloc[i]
                all_rows.append({
                    "date":      row["date"],
                    "actual":    float(row[target]),
                    "predicted": float(preds_np[0, pred_offset + i]),
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



# try this out!
# from data_frame_builder import DataFrameBuilder

# path = "/Users/minna/Code/FS26/AML/aml2026-group-3"
# dfb  = DataFrameBuilder(path)
# df   = dfb.process_data()

# splits, holdout = dfb.generate_split(df)

# for s in splits:
#     tr, te = s["train"], s["test"]
#     print(f"Fold {s['fold']}: train [{tr['date'].min().date()} – {tr['date'].max().date()}] ({len(tr)} rows) | "
#           f"test [{te['date'].min().date()} – {te['date'].max().date()}] ({len(te)} rows)")

# tftr = TFTRunner(dfb)
# ckpt = tftr.run(splits=splits, target="CPI")
# print(f"Best checkpoint: {ckpt}")

