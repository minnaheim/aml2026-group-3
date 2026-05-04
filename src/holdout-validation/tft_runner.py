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
        # print(train_raw.isna().sum())
        train_df  = self._add_tft_vars(train_raw)
        training_ds, train_dl, val_dl = self.create_tft_dataset(train_df, target, batch_size)
        return self.train_tft(training_ds, train_dl, val_dl, use_tqdm, use_wandb, device)


# try this out! 

# from data_frame_builder import DataFrameBuilder 

# # from data_frame_builder
# path = "/Users/minna/Code/FS26/AML/aml2026-group-3"
# dfb = DataFrameBuilder(path)
# df = dfb.process_data()

# splits, holdout = dfb.generate_split(df)

# for s in splits:
#     tr, te = s["train"], s["test"]
#     print(f"Fold {s['fold']}: train [{tr['date'].min().date()} – {tr['date'].max().date()}] ({len(tr)} rows) | "
#           f"test [{te['date'].min().date()} – {te['date'].max().date()}] ({len(te)} rows)")

# # from tft-runner
# tftr = TFTRunner(dfb)
# # run training
# tftr.run(splits = splits, target = "CPI") 

