# --------- setup -------------
import pytorch_forecasting
import warnings
from tqdm import tqdm
warnings.filterwarnings("ignore")
import pandas as pd
import os
import matplotlib.pyplot as plt
import holidays
# different than original example
import lightning.pytorch as pl
import torch
DEVICE = 'mps' if torch.backends.mps.is_available() else 'cpu'
from pytorch_forecasting import Baseline, TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.data import GroupNormalizer, NaNLabelEncoder
from pytorch_forecasting.metrics import SMAPE, PoissonLoss, QuantileLoss
from pytorch_forecasting.models.temporal_fusion_transformer.tuning import optimize_hyperparameters
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor
# using weights and biases to visualise training
from lightning.pytorch.loggers import WandbLogger
wandb_logger = WandbLogger(project="tft", name="tft-longformat")

# --------- data processing -------------
daily_path = '/Users/minna/Code/FS26/AML/aml2026-group-3/data/macro-vars-daily.csv'
df_daily = pd.read_csv(daily_path)
df_daily = df_daily.rename(columns = {'Unnamed: 0':'date'})
df_daily['date'] = pd.to_datetime(df_daily['date'])
df_daily = df_daily.drop(columns=["SOFR", "T10Y2Y", "EUR"]) # remove shorter vars: SOFR, T10Y2Y, EUR
print(df_daily.tail(20))

qrtly_path = '/Users/minna/Code/FS26/AML/aml2026-group-3/data/macro-vars-quarterly.csv'
df_quartly = pd.read_csv(qrtly_path)
df_quartly = df_quartly.rename(columns = {'Unnamed: 0':'date'})
df_quartly['date'] = pd.to_datetime(df_quartly['date'])
print(df_quartly.tail(10))

# os.getcwd()
monthly_path = '/Users/minna/Code/FS26/AML/aml2026-group-3/data/macro-vars-monthly.csv'
df_monthly = pd.read_csv(monthly_path)
df_monthly = df_monthly.rename(columns = {'Unnamed: 0':'date'})
# remove columns which are too short 
df_monthly = df_monthly.drop(columns=["PCEPI", "JTSJOL", "UMCSENT"])

df_monthly['date'] = pd.to_datetime(df_monthly['date'])
# print(df_monthly.tail(10))

# merge 
df = pd.merge(df_monthly, df_quartly, on='date', how='left')
df = pd.merge(df_daily, df, on='date', how='left') # TODO: only 2 at a time?
# here you can see gdp has a stronger publication lag than the motnhly vars
# print(df.tail(20)) 

# forward-fill monthly/quarterly vars: repeat last published value until updated
monthly_quarterly_cols = ["CPI", "PAYEMS", "INDPRO", "UNRATE", "GDP"]
df[monthly_quarterly_cols] = df[monthly_quarterly_cols].ffill()
print(f"NAs after ffill:\n{df[monthly_quarterly_cols].isna().sum()}")
print(f"\nDate range: {df['date'].min()} to {df['date'].max()}, {len(df)} rows")

# basic plot to inspect length
df.plot(x='date', y=df.columns.drop("date"), subplots=True, figsize=(10, 10))


# ------------- setup TFT -------------

# adding time varying known covariates
us_holidays = holidays.US()

df['day_of_week'] = df['date'].dt.dayofweek
df['week_of_year'] = df['date'].dt.isocalendar().week.astype(int)
df['month'] = df['date'].dt.month
df['is_holiday'] = df['date'].dt.date.apply(lambda d: int(d in us_holidays))

df['series_id'] = 'macro'

# precompute monthly lags: lag=N means the value from N calendar months ago
# use merge_asof so the offset is in months, not rows (which would be meaningless on ffilled daily data)
lag_vars = ["CPI", "PAYEMS", "INDPRO", "UNRATE", "GDP"]
lag_periods = [1, 2, 6, 12] # lag not the actual day but the quarter/month -> else the lag_1 is 1 day for monthly & quarterly vars

source = df[['date'] + lag_vars].sort_values('date').reset_index(drop=True)
for col in lag_vars:
    for lag in lag_periods:
        lookup = pd.DataFrame({
            'date': df['date'] - pd.DateOffset(months=lag),
            '_orig': range(len(df))
        }).sort_values('date').reset_index(drop=True)
        merged = pd.merge_asof(lookup, source[['date', col]], on='date', direction='backward')
        df[f"{col}_lag_{lag}"] = merged.sort_values('_orig')[col].values

# TODO: maybe don't go up to lag 12... then discard a lot of data
# drop rows where 12-month lag is undefined (first ~12 months of series)
df = df.dropna().reset_index(drop=True)

# create time_idx
df = (df.merge((df[['date']].drop_duplicates(ignore_index=True)
.rename_axis('time_idx')).reset_index(), on=['date']))

print(df.tail(10))
print(f"\nRemaining NAs:\n{df.isna().sum()[df.isna().sum() > 0]}")

# split into test, train
tr_len = round(len(df) * 0.8)

train = df.iloc[:tr_len]
test = df.iloc[tr_len:]
print(train.tail()) # goes until 2004

print(train.columns)


meta_raw = pd.read_csv('/Users/minna/Code/FS26/AML/aml2026-group-3/data/metadata-macro.csv', index_col=1)
meta_raw = meta_raw.rename(index={'CPIAUCSL': 'CPI', "DEXUSUK": "GBP", "DEXJPUS":"YEN", "DFF":"FFR"})

# years of history through end of training — avoids 0 for GBP/YEN whose series start coincides with train_start
train_end = train['date'].max()
meta_raw['meta_years_of_history'] = meta_raw['observation_start'].apply(
    lambda s: max(0.0, (train_end - pd.to_datetime(s)).days / 365.25)
)
# TODO: include metadata for all vars, not just target??
# add frequency too, now that we have quarterly (and differentiate between 7day daily, and Daily 5 days)
meta = meta_raw[['popularity', 'units_short', 'meta_years_of_history', 'frequency_short']].rename(
    columns={'popularity': 'meta_popularity', 'units_short': 'meta_units', 'frequency_short':'meta_frequency'}
).loc[["CPI", "PAYEMS", "INDPRO", "UNRATE", "GDP"]]

print(meta)


# -------------- training tft ----------------

# adjust to match daily length
max_encoder_length = 48*30  # 4 years lookback
max_prediction_length = 12*30  # 12 months forecast

lag_vars = ["CPI", "PAYEMS", "INDPRO", "UNRATE", "GDP"]
lag_periods = [1, 2, 6, 12]
lagged_cols = [f"{col}_lag_{lag}" for col in lag_vars for lag in lag_periods]

# only include target metadata, other one was too bad.
train_aug = train.assign(**meta.loc["INDPRO"].to_dict())

training = TimeSeriesDataSet(
    data=train_aug,
    time_idx="time_idx",
    target="INDPRO",
    group_ids=["series_id"],
    max_encoder_length=max_encoder_length,
    max_prediction_length=max_prediction_length,
    static_categoricals=["series_id", "meta_units", "meta_frequency"],
    static_reals=["meta_popularity", "meta_years_of_history"],
    time_varying_known_reals=["time_idx", "day_of_week", "week_of_year", "month", "is_holiday"],
    time_varying_known_categoricals=[],
    time_varying_unknown_reals=["CPI", "PAYEMS", "INDPRO", "UNRATE", "GDP"] + lagged_cols,
    add_relative_time_idx=True,
    add_target_scales=True,
    add_encoder_length=True,
    allow_missing_timesteps=True,
)

validation = TimeSeriesDataSet.from_dataset(training, train_aug, predict=True, stop_randomization=True)

batch_size = 128
train_dataloader = training.to_dataloader(train=True, batch_size=batch_size, num_workers=0)
val_dataloader = validation.to_dataloader(train=False, batch_size=batch_size * 10, num_workers=0)

print(train_aug.columns)
print(train_aug['meta_units'])

# setting hyperparams
PATIENCE = 5
MAX_EPOCHS = 40
LEARNING_RATE = 0.03
OPTUNA = False # hyperparam opt


from lightning.pytorch.callbacks.progress import TQDMProgressBar

early_stop_callback = EarlyStopping(monitor="train_loss", min_delta=1e-2, patience=PATIENCE, verbose=False, mode="min")
lr_logger = LearningRateMonitor()  # log the learning rate

trainer = pl.Trainer(
    max_epochs=MAX_EPOCHS,
    accelerator="auto",  # uses MPS on Apple Silicon, CPU otherwise
    devices=1,
    enable_model_summary=True,
    gradient_clip_val=0.25,
    limit_train_batches=5,
    callbacks=[lr_logger, early_stop_callback, TQDMProgressBar(refresh_rate=1)],
    logger=wandb_logger
)

tft = TemporalFusionTransformer.from_dataset(
    training,
    learning_rate=LEARNING_RATE,
    lstm_layers=2,
    hidden_size=16,
    attention_head_size=2,
    dropout=0.2,
    hidden_continuous_size=8,
    output_size=1,  # 7 quantiles by default
    loss=SMAPE(),
    log_interval=10,
    reduce_on_plateau_patience=4
)

tft.to(DEVICE)
print(f"Number of parameters in network: {tft.size()/1e3:.1f}k")

# add tqdm for progress report
trainer.fit(
    tft,
    train_dataloaders=train_dataloader,
    val_dataloaders=val_dataloader,
)


best_model_path = trainer.checkpoint_callback.best_model_path
best_tft = TemporalFusionTransformer.load_from_checkpoint(best_model_path)

actuals = torch.cat([y[0] for x, y in iter(val_dataloader)])
predictions = best_tft.predict(val_dataloader, mode="prediction")
# print(predictions[:10])
# print(predictions.shape)

raw_output = best_tft.predict(val_dataloader, mode="raw", return_x=True)
# this is how to unpack it 
raw_predictions = raw_output.output
x = raw_output.x

sm = SMAPE()
print(f"Validation median SMAPE loss: {sm.loss(actuals, predictions).mean(axis=1).median().item()}")