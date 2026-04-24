#!/usr/bin/env python
# coding: utf-8

# # ARIMA
# Macro-only benchmark: Box-Jenkins ARIMA models use autoregressive lags, differencing, and moving-average terms to model persistent macroeconomic dynamics. In this project, ARIMA serves as a transparent classical baseline for GDP growth and inflation forecasts, allowing us to test whether text-enhanced TFT improves beyond a strong traditional time-series model. The ARIMA benchmark was used as an evaluation metric for the Laborda et al. project, listed below.
# 
# Reference: Box, G.; Jenkins, G.M. Time Series Analysis; Forecasting and Control; Holden-Day: San Francisco, CA, USA, 1970.

# set-up
import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller
import matplotlib.pyplot as plt
from pmdarima import auto_arima
from pathlib import Path

import os
#os.chdir(os.path.dirname(os.path.abspath(__file__)) + "/..")
os.chdir(r"C:\Users\annaz\OneDrive\Dokumente\Studium\UZH_Master\2026FS\Advanced Machine Learning\Practical_Assignment\aml2026-group-3")
print(os.getcwd())


# define and create folder for results of arima
arima_dir = Path("out/arima")
arima_dir.mkdir(parents=True, exist_ok=True)



# load data
path_monthly = "data/macro-vars-monthly.csv"
path_daily = "data/macro-vars-daily.csv"
path_weekly = "data/macro-vars-weekly.csv"
path_quarterly = "data/macro-vars-quarterly.csv"

df_monthly = pd.read_csv(path_monthly)
df_daily = pd.read_csv(path_daily)
df_weekly = pd.read_csv(path_weekly)
df_quarterly = pd.read_csv(path_quarterly)

df_businessdays = df_daily[['Unnamed: 0', 'EUR', 'GBP', 'YEN', 'SOFR', 'T10Y2Y']].copy()
# drop weekends:
#cols_to_check = df_businessdays.columns.difference(['Unnamed: 0'])
#df_businessdays = df_businessdays.dropna(subset=cols_to_check, how='all')

df_7days = df_daily[['Unnamed: 0', 'FFR']].copy()


# put into dictionary
dataframes = {
    "Monthly": df_monthly,
    "Weekly": df_weekly,
    "Daily_5day": df_businessdays,
    "Daily_7day": df_7days,
    "Quarterly": df_quarterly
}

# get frequency mapping
freq_map = {'Monthly': 'MS', 'Daily_5day': 'B', 'Daily_7day': 'D', 'Weekly': 'W-WED', 'Quarterly': 'QS'}
# W-WED since weekly data recorded on wednesdays
# B for business day daily data since we care about business days not 7-day week!

for name, df in dataframes.items():
    # rename first column into date column
    if 'Unnamed: 0' in df.columns:
        df.rename(columns={'Unnamed: 0': 'Date'}, inplace=True)
    
    # set the index
    if 'Date' in df.columns:
        df.set_index('Date', inplace=True)
    
    # index as date format
    df.index = pd.to_datetime(df.index)
    
    # filter date range
    dataframes[name] = df.loc[df.index <= '2026-02-01'].copy()
    
    # set frequency, depending on df
    dataframes[name] = dataframes[name].asfreq(freq_map[name])
    
    print(f"{name} processed. Shape: {dataframes[name].shape}")

# initialize
adf_summary = []

# first: stationarity? likely not with trend variables
for name, df in dataframes.items():
    for col in df.columns:
        # Drop NaNs for each specific column to avoid errors
        result = adfuller(df[col].dropna())
        
        series = df[col].dropna()
        if len(series) > 5:
            result = adfuller(series)
            p_val = result[1]
            adf_summary.append({
                'Frequency': name,
                'Variable': col,
                'p-value': round(p_val, 4),
                'Stationary': 'Yes' if p_val < 0.05 else 'No'
            })
        
        
# view as a clean summary table
results_df = pd.DataFrame(adf_summary)
print(results_df.sort_values(by=['Frequency', 'Stationary']))

# stationary: YES, T10Y2Y, FFR

log_cols = ['CPI', 'PCEPI', 'JTSJOL', 'PAYEMS', 'INDPRO', 'GDP', 'EUR', 'GBP', 'WALCL'] # strictly positive, makes sense to log since want growth rate
diff_only = ['UMCSENT', 'SOFR'] # difference without log (already an index)

# dictionary to store different stationarity results
stationary_results = []


for freq_name, df in dataframes.items():
    print(f"\n--- Testing Transformations for {freq_name} ---")
    
    for col in df.columns:
        series = df[col].dropna()
        
        # log difference
        if col in log_cols:
            transformed = np.log(series).diff().dropna()
            label = "log_diff"
            
        # simple difference only
        elif col in diff_only:
            transformed = series.diff().dropna()
            label = "diff only"
            
        # if stationary, skip
        else:
            continue

        # again, adf to check for stationarity
        if len(transformed) > 5:
            pval = adfuller(transformed)[1]
            is_stationary = "Yes" if pval < 0.05 else "No"
            
            print(f"{col} ({label}): p = {pval:.6f} | Stationary: {is_stationary}")
            
            stationary_results.append({
                'Frequency': freq_name,
                'Variable': col,
                'Method': label,
                'p-value': round(pval, 6),
                'Stationary': is_stationary
            })

# look at summary
transformation_summary = pd.DataFrame(stationary_results)

# SOFR still isn't stationary
# this is relatively unsurprising
# it's very volatile and the time horizon is fairly long
# check if second differencing works
# interpretation = how much does the change in the interest rate (SOFR) accelerate?

df_businessdays['SOFR_diff2'] = df_businessdays['SOFR'].diff().diff().dropna()

# check adf again
sofr_2_clean = df_businessdays['SOFR_diff2'].dropna()
result = adfuller(sofr_2_clean)

print(f"SOFR Second Difference p-value: {result[1]:.6f}")
# yes, looks good now!

# overwrite the differencing variable
first_diff = ['UMCSENT'] # difference without log (already an index)
second_diff = ['SOFR'] # second difference

# apply the transformations we defined above
stationary_dfs = {}

for name, df in dataframes.items():
    df_stat = df.copy()
    
    for col in df_stat.columns:
        # log-differencing
        if col in log_cols:
            df_stat[col] = np.log(df_stat[col]).diff()
            
        # first difference
        elif col in first_diff:
            df_stat[col] = df_stat[col].diff()
            
                    
        # second-differences
        elif col in second_diff:
            df_stat[col] = df_stat[col].diff().diff()
            
    # do NOT drop nans
    # for ARIMA and AR, we don't need complete panel of variables which we would achieve with this
    # just complete time series
    stationary_dfs[name] = df_stat


#------------------------------------------------------------------
# ARIMA: Pseudo-Forecasting
#------------------------------------------------------------------


# set-up
forecast_config = {
    'Monthly': 12,    # 1 year
    'Weekly': 8,      # 2 months
    'Daily_5day': 20, # 1 month in business days
    'Daily_7day': 30, # 1 month incl weekends
    'Quarterly': 4    # 1 year
}

seasonal_periods = {
    'Monthly': 12, 
    'Weekly': 52, 
    'Daily_5day': 5,
    'Daily_7day': 7, 
    'Quarterly': 4
}

all_models = {}
all_forecasts = {}
all_conf_ints = {}

for freq_name, df in stationary_dfs.items():
    n_steps = forecast_config[freq_name]
    m = seasonal_periods[freq_name]
    
    print(f"Fitting models for {freq_name} data (Holding out {n_steps} periods)...")
    
    # frequency-specific storage
    all_models[freq_name] = {}
    all_forecasts[freq_name] = {}
    all_conf_ints[freq_name] = {}

    for col in df.columns:
        series = df[col].asfreq(freq_map[freq_name])
        
        # train/test split
        train = series.iloc[:-n_steps]
        test = series.iloc[-n_steps:] # for validation
        
        # define and fit
        # we use m from our seasonal_periods map to account for differing frequencies and seasonalities
        model = SARIMAX(train,
                        order=(1, 0, 1),
                        seasonal_order=(0, 0, 1, m),
                        enforce_stationarity=False,
                        enforce_invertibility=False)
        
        res = model.fit(disp=False)
        all_models[freq_name][col] = res
        
        # Get Forecast
        forecast_obj = res.get_forecast(steps=n_steps)
        all_forecasts[freq_name][col] = forecast_obj.predicted_mean
        all_conf_ints[freq_name][col] = forecast_obj.conf_int()

print("All models fitted successfully.")



# --- Forecast errors ---
evaluation_results = []

for freq_name, df in stationary_dfs.items():
    n_steps = forecast_config[freq_name]
    
    for col in df.columns:
        actual = df[col].iloc[-n_steps:] # last n_steps of stationary data
        pred = all_forecasts[freq_name][col] #
        
        # errors
        error = pred - actual
        mae = error.abs().mean()
        rmse = (error ** 2).mean() ** 0.5
        
        # normalized RMSE
        # using the standard deviation of the entire data series
        data_std = df[col].std()
        nrmse = rmse / data_std if data_std != 0 else np.nan
        
        evaluation_results.append({
            'Frequency': freq_name,
            'Variable': col,
            'MAE': round(mae, 6),
            'RMSE': round(rmse, 6),
            'NRMSE': round(nrmse, 6)
        })

# summary table
error_report = pd.DataFrame(evaluation_results)

# sort by NRMSE to see how models are ranked
print(error_report.sort_values(by='NRMSE'))


# --- Plots ---
for freq_name, df in stationary_dfs.items():
    cols = df.columns
    n_vars = len(cols)
    n_steps = forecast_config[freq_name]
    
    # 1. Setup figure
    n_cols = 2
    n_rows = int(np.ceil(n_vars / n_cols))
    
    # Use squeeze=False to ensure axes is ALWAYS a 2D array, 
    # which makes flattening predictable
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, n_rows * 4), squeeze=False)
    fig.suptitle(f"Forecasts for {freq_name} Variables", fontsize=16, fontweight='bold')
    
    # Flatten the 2D array into a 1D list of axes
    flat_axes = axes.flatten()

    for i, col in enumerate(cols):
        ax = flat_axes[i]
        
        # Extract Data
        full_series = df[col]
        train_slice = full_series.iloc[:-n_steps]
        test_slice = full_series.iloc[-n_steps:]
        
        # Get predictions from our nested dicts
        forecast_series = all_forecasts[freq_name][col]
        ci = all_conf_ints[freq_name][col]
        
        # Calculate local RMSE
        rmse_val = (((forecast_series - test_slice) ** 2).mean()) ** 0.5

        # 2. Plotting (using the axis 'ax' explicitly)
        context_window = 60 if freq_name == 'Daily' else 24
        
        # Plot history
        ax.plot(train_slice.index[-context_window:], train_slice.values[-context_window:], 
                label='Train', color='steelblue')
        
        # Plot actuals
        ax.plot(test_slice.index, test_slice.values, 
                label='Actual', color='black', linewidth=1.5)
        
        # Plot forecast
        ax.plot(forecast_series.index, forecast_series.values, 
                label='Forecast', color='crimson', linestyle='--')

        # Confidence Interval
        ax.fill_between(ci.index, ci.iloc[:, 0], ci.iloc[:, 1], alpha=0.15, color='crimson')

        # Formatting
        ax.set_title(f"{col} | RMSE: {rmse_val:.4f}", fontsize=11)
        ax.legend(loc='upper left', fontsize=8)
        ax.grid(alpha=0.2)

    # 3. Hide unused subplots in the grid
    for j in range(i + 1, len(flat_axes)):
        flat_axes[j].set_visible(False)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.close()


#------------------------------------------------------------------
# Rolling Window Approach
#------------------------------------------------------------------

# DEFINE helpers

def get_folds(df, n_folds, max_horizon=12):
  """Generate train/test splits, each test window = max_horizon months"""
  n = len(df)
  test_size = max_horizon
  # space folds evenly across the series
  # here, we luckily don't have to worry about data frequency since i am only using monthly data
  # for now!
  fold_ends = np.linspace(n - n_folds * test_size, n - test_size, n_folds, dtype=int)

  folds = []
  for end in fold_ends:
      train = df.iloc[:end]
      test = df.iloc[end:end + test_size]
      folds.append((train, test))
  return folds


def fit_and_forecast(train_series, horizon, m):
  train_series = train_series.dropna()
  model = SARIMAX(train_series,
                  order=(1, 0, 1),
                  seasonal_order=(0, 0, 1, m),
                  enforce_stationarity=False,
                  enforce_invertibility=False)
  result = model.fit(maxiter = 250,
                     disp=False)
  forecast = result.get_forecast(steps=horizon)
  return forecast.predicted_mean


def compute_metrics(actual, predicted):
  errors = actual.values - predicted.values
  mae  = np.mean(np.abs(errors))
  rmse = np.sqrt(np.mean(errors ** 2))
  # mask zeros to avoid division issues in MAPE
  mask = actual.values != 0
  mape = np.mean(np.abs(errors[mask] / actual.values[mask])) * 100
  return {"MAE": mae, "RMSE": rmse, "MAPE": mape}

# helper: since our data is often transformed, transform it back to calculate MAPE
def invert_transform(col, forecasted_diff, last_known_value, is_log=False):
    if is_log:
        last_log = np.log(last_known_value)
        return np.exp(last_log + forecasted_diff.cumsum())
    else:
        return last_known_value + forecasted_diff.cumsum()


#------------------------------------------------------------------



all_cv_results = {}
all_cv_forecasts = {}

forecast_horizons = {
    freq_name: list(range(1, n_steps + 1))
    for freq_name, n_steps in forecast_config.items()
}

for freq_name, df_transformed in stationary_dfs.items():
    df_orig = dataframes[freq_name]
    n_steps = forecast_config[freq_name]
    m = seasonal_periods[freq_name]
    horizons = forecast_horizons[freq_name]
    
    df_transformed.index.freq = freq_map[freq_name]

    folds_trans = get_folds(df_transformed, n_folds=5, max_horizon=n_steps)
    folds_orig  = get_folds(df_orig,         n_folds=5, max_horizon=n_steps)

    freq_metrics   = []
    # per-column, per-horizon store: {col: {h: [{'forecast': ..., 'actual': ...}, ...]}}
    freq_forecasts = {col: {h: [] for h in horizons} for col in df_transformed.columns}

    for col in df_transformed.columns:
        is_log         = col in log_cols
        is_second_diff = col in second_diff

        # per-horizon metric accumulators
        col_results = {h: [] for h in horizons}  # list of metric dicts per fold

        for i in range(len(folds_trans)):
            df_train_t, df_test_t = folds_trans[i]
            df_train_o, df_test_o = folds_orig[i]

            train_t = df_train_t[col].asfreq(freq_map[freq_name])
            train_o = df_train_o[col]
            test_o  = df_test_o[col]

            # forecast at max horizon; slice per h below
            pred_diffs = fit_and_forecast(train_t, n_steps, m)

            last_val = train_o.dropna().iloc[-1]

            if is_second_diff:
                last_diff        = train_o.diff().dropna().iloc[-1]
                reconstructed    = last_diff + pred_diffs.cumsum()
                pred_inverted    = last_val + reconstructed.cumsum()
            else:
                pred_inverted = invert_transform(col, pred_diffs, last_val, is_log=is_log)
                
            pred_inverted.index = test_o.index[:len(pred_inverted)]

            # iterate over horizons, slicing the full forecast
            for h in horizons:
                actual_h   = test_o.iloc[:h]
                forecast_h = pred_inverted.iloc[:h]

                # NaN mask (mirrors old version)
                mask           = actual_h.notna()
                actual_clean   = actual_h[mask]
                forecast_clean = forecast_h[mask]

                if len(actual_clean) == 0:
                    continue

                # align index (mirrors old version)
                forecast_clean.index = actual_clean.index

                metrics = compute_metrics(actual_clean, forecast_clean)
                col_results[h].append(metrics)               # raw per-fold, per-horizon

                freq_forecasts[col][h].append({              # store for plotting
                    'forecast': forecast_clean,
                    'actual':   actual_clean,
                })

        # average across folds per horizon
        for h in horizons:
            if not col_results[h]:
                continue
            freq_metrics.append({
                'Variable': col,
                'Horizon':  h,
                'CV_MAE':   np.mean([m['MAE']  for m in col_results[h]]),
                'CV_RMSE':  np.mean([m['RMSE'] for m in col_results[h]]),
                'CV_MAPE':  np.mean([m['MAPE'] for m in col_results[h]]),
            })

    all_cv_results[freq_name]   = pd.DataFrame(freq_metrics)
    all_cv_forecasts[freq_name] = freq_forecasts
    
    
    
    
mean_levels = {
    freq_name: dataframes[freq_name].mean()
    for freq_name in dataframes
}

for freq_name, results_df in all_cv_results.items():
    levels = mean_levels[freq_name]
    
    results_df["NRMSE"] = results_df.apply(
        lambda row: row["CV_RMSE"] / levels[row["Variable"]], axis=1
    )
    
    print(f"\n=== {freq_name} ===")
    print(
        results_df
        .pivot(index="Variable", columns="Horizon", values=["CV_MAE", "CV_RMSE", "CV_MAPE", "NRMSE"])
    )
    
    
plot_horizon = {
    'Monthly':    12,
    'Weekly':      8,
    'Daily_5day': 20,
    'Daily_7day': 30,
    'Quarterly':   4,
}

x = {
    'Monthly':    72,   # 6 years
    'Weekly':     104,  # 2 years
    'Daily_5day': 260,  # ~1 year business days
    'Daily_7day': 365,  # 1 year
    'Quarterly':  20,   # 5 years
}

key_horizons = {
    'Monthly':    [1, 6, 12],
    'Weekly':     [1, 4, 8],
    'Daily_5day': [1, 10, 20],
    'Daily_7day': [1, 15, 30],
    'Quarterly':  [1, 2, 4],
}

for freq_name, df_transformed in stationary_dfs.items():
    df_orig     = dataframes[freq_name]
    results_df  = all_cv_results[freq_name]
    forecasts   = all_cv_forecasts[freq_name]
    h_plot      = plot_horizon[freq_name]
    tail_n      = x[freq_name]

    cols   = df_orig.columns
    n_cols = 2
    n_rows = int(np.ceil(len(cols) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, n_rows * 4))
    axes = axes.flatten()
    n_folds = n_folds = (max(
        len(entries)
        for col_dict in forecasts.values()
        for entries in col_dict.values()
        if entries))  # skip empty lists  # infer from data
    
    colors  = plt.cm.RdYlGn(np.linspace(0.2, 0.8, n_folds))

    for i, col in enumerate(cols):
        ax = axes[i]
        actual_subset = df_orig[col].tail(tail_n)
        ax.plot(actual_subset.index, actual_subset.values,
                color='black', linewidth=1, label='Actual', alpha=0.6, zorder=2)

        for fold_i, entry in enumerate(forecasts[col][h_plot]):
            color = colors[fold_i % len(colors)]
            ax.plot(entry['forecast'], color=color, linewidth=1.5,
                    linestyle='--', alpha=0.9, label=f'Fold {fold_i+1}')
            ax.plot(entry['actual'],   color=color, linewidth=1.5,
                    alpha=0.5)

        ax.set_xlim(actual_subset.index.min(), actual_subset.index.max())

        # NRMSE in title — note updated column names
        nrmse_row = results_df[results_df['Variable'] == col]
        nrmse_1   = nrmse_row[nrmse_row['Horizon'] == 1][      'NRMSE'].values
        nrmse_h   = nrmse_row[nrmse_row['Horizon'] == h_plot]['NRMSE'].values
        title = f"{col}  |  NRMSE h=1: {nrmse_1[0]:.3f}  h={h_plot}: {nrmse_h[0]:.3f}" \
                if len(nrmse_1) and len(nrmse_h) else col

        ax.set_title(title, fontsize=9)
        ax.legend(fontsize=7, ncol=3)
        ax.grid(alpha=0.3)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(f"CV Forecasts — {freq_name}", fontsize=12, y=1.01)
    plt.tight_layout()
    plt.close()


#------------------------------------------------------------------
# VALIDATION
#------------------------------------------------------------------
# Validation: Choosing the optimal ARIMA(p,d,q)
# 
# *   p = degree of autoregressivity
# *   d = degree of integration => 0 for us since we manually use log differencing to keep track of data transformations for other models. This improves the computational speed of auto_arima (to find optimal (p,d,q) per variable) substantially!
# *   q = degree of moving average
# 

n_test = {
    'Monthly':    24,   # 2 years
    'Weekly':     16,   # 4 months
    'Daily_5day': 60,   # 3 months business days
    'Daily_7day': 90,   # 3 months
    'Quarterly':   8,   # 2 years
}

n_validation = {
    'Monthly':    24,
    'Weekly':     16,
    'Daily_5day': 60,
    'Daily_7day': 90,
    'Quarterly':   8,
}

train_val   = {}
test_sets   = {}
train_val_orig = {}
test_orig      = {}

for freq_name in dataframes:
    df_t = stationary_dfs[freq_name]
    df_o = dataframes[freq_name]
    nt   = n_test[freq_name]

    train_val[freq_name]      = df_t.iloc[:-nt]
    test_sets[freq_name]      = df_t.iloc[-nt:]
    train_val_orig[freq_name] = df_o.iloc[:-nt]
    test_orig[freq_name]      = df_o.iloc[-nt:]
    
# define max seasonality that we allow, otherwise, state space matrix from Kalman filter too large for memory
m_arima = {
    'Monthly':    12,
    'Weekly':      4,  # use monthly seasonality instead of m=52
    'Daily_5day':  5,
    'Daily_7day':  7,
    'Quarterly':   4,
}

# order selection via auto_arima on train, evaluated on validation
best_orders = {}

for freq_name in dataframes:
    df_t = stationary_dfs[freq_name]
    m    = m_arima[freq_name]
    nv   = n_validation[freq_name]
    
    # restrict seasonal terms for high-frequency data, again for memory
    start_P = 0 if freq_name in ('Weekly', 'Daily_5day', 'Daily_7day') else 1
    start_Q = 0 if freq_name in ('Weekly', 'Daily_5day', 'Daily_7day') else 1
    max_P = 0 if freq_name in ('Weekly', 'Daily_5day', 'Daily_7day') else 2
    max_Q = 0 if freq_name in ('Weekly', 'Daily_5day', 'Daily_7day') else 2
    
    best_orders[freq_name] = {}
    
    for col in df_t.columns:
        train_series = train_val[freq_name].iloc[:-nv][col].dropna()
        model = auto_arima(train_series,
                           seasonal=True, m=m,
                           stepwise=True,
                           information_criterion='aic',
                           max_p=3, max_q=3,
                           start_P=start_P, start_Q=start_Q,
                           max_P=max_P, max_Q=max_Q,
                           d=0, D=0,
                           trace=False,
                           error_action='ignore',
                           suppress_warnings=True)
        best_orders[freq_name][col] = {
            'order':          model.order,
            'seasonal_order': model.seasonal_order,
        }
        print(f"[{freq_name}] {col}: {model.order} x {model.seasonal_order}")
        
        
#------------------------------------------------------------------
# update helper functions from above
def get_folds(df, n_folds, max_horizon):
    """Generate train/test splits, each test window = max_horizon months"""
    n = len(df)
    test_size = max_horizon
    fold_ends = np.linspace(n - n_folds * test_size, n - test_size, n_folds, dtype=int)
    folds = []
    for end in fold_ends:
        train = df.iloc[:end]
        test  = df.iloc[end:end + test_size]
        folds.append((train, test))
    return folds


def fit_and_forecast(train_series, horizon, freq_name, col):
    train_series   = train_series.dropna()
    order          = best_orders[freq_name][col]['order']
    seasonal_order = best_orders[freq_name][col]['seasonal_order']
    model = SARIMAX(train_series,
                    order=order,
                    seasonal_order=seasonal_order,
                    enforce_stationarity=False,
                    enforce_invertibility=False)
    result   = model.fit(maxiter=250, disp=False)
    forecast = result.get_forecast(steps=horizon)
    return forecast.predicted_mean


def compute_metrics(actual, predicted):
    errors = actual.values - predicted.values
    mae    = np.mean(np.abs(errors))
    rmse   = np.sqrt(np.mean(errors ** 2))
    mask   = actual.values != 0
    mape   = np.mean(np.abs(errors[mask] / actual.values[mask])) * 100
    return {"MAE": mae, "RMSE": rmse, "MAPE": mape}

# to go back to original data value (so, before log differencing)
def invert_transform(col, forecasted_diff, last_known_value, is_log=False):
    if is_log:
        last_log = np.log(last_known_value)
        return np.exp(last_log + forecasted_diff.cumsum())
    else:
        return last_known_value + forecasted_diff.cumsum()

#------------------------------------------------------------------

all_cv_results   = {}
all_cv_forecasts = {}

for freq_name in dataframes:
    df_t   = stationary_dfs[freq_name]
    df_o   = dataframes[freq_name]
    n_steps = forecast_config[freq_name]
    m       = seasonal_periods[freq_name]
    horizons = key_horizons[freq_name]
    n_folds  = 5

    folds_transformed = get_folds(train_val[freq_name],      n_folds=n_folds, max_horizon=n_steps)
    folds_original    = get_folds(train_val_orig[freq_name], n_folds=n_folds, max_horizon=n_steps)

    cv_results   = {col: {h: [] for h in horizons} for col in df_t.columns}
    cv_forecasts = {col: {h: [] for h in horizons} for col in df_t.columns}

    for fold_i, ((train, test), (train_orig, test_orig)) in enumerate(zip(folds_transformed, folds_original)):
        print(f"[{freq_name}] Fold {fold_i+1}: train up to {train.index[-1].date()}, "
              f"test {test.index[0].date()} → {test.index[-1].date()}")

        for col in df_t.columns:
            for h in horizons:
                train_col = train[col].asfreq(freq_map[freq_name])
                forecast  = fit_and_forecast(train_col, h, freq_name, col)

                # invert transform
                is_log         = col in log_cols
                is_second_diff = col in second_diff

                if is_second_diff:
                    last_val       = train_orig[col].dropna().iloc[-1]
                    last_diff      = train_orig[col].diff().dropna().iloc[-1]
                    reconstructed  = last_diff + forecast.cumsum()
                    forecast_levels = last_val + reconstructed.cumsum()
                    actual_levels   = test_orig[col].iloc[:h]
                elif is_log:
                    last_val        = train_orig[col].iloc[-1]
                    forecast_levels = invert_transform(col, forecast, last_val, is_log=True)
                    actual_levels   = test_orig[col].iloc[:h]
                elif col in diff_only:
                    last_val        = train_orig[col].iloc[-1]
                    forecast_levels = invert_transform(col, forecast, last_val, is_log=False)
                    actual_levels   = test_orig[col].iloc[:h]
                else:
                    forecast_levels = forecast
                    actual_levels   = test[col].iloc[:h]

                # realign index before NaN mask
                forecast_levels.index = test_orig[col].index[:len(forecast_levels)]

                mask           = actual_levels.notna()
                actual_clean   = actual_levels[mask]
                forecast_clean = forecast_levels[mask]

                if len(actual_clean) == 0:
                    continue

                forecast_clean.index = actual_clean.index
                metrics = compute_metrics(actual_clean, forecast_clean)
                cv_results[col][h].append(metrics)
                cv_forecasts[col][h].append({'forecast': forecast_clean, 'actual': actual_clean})

    all_cv_results[freq_name]   = cv_results
    all_cv_forecasts[freq_name] = cv_forecasts
    
    
    
mean_levels = {
    freq_name: dataframes[freq_name].mean()
    for freq_name in dataframes
}

all_results_df = {}

for freq_name in dataframes:
    levels   = mean_levels[freq_name]
    horizons = key_horizons[freq_name]
    cv_res   = all_cv_results[freq_name]

    rows = []
    for col, h_dict in cv_res.items():
        for h, fold_metrics in h_dict.items():
            if h not in horizons or len(fold_metrics) == 0:
                continue
            mean_rmse = np.mean([m["RMSE"] for m in fold_metrics])
            rows.append({
                "variable":  col,
                "horizon":   h,
                "mean_RMSE": mean_rmse,
                "mean_MAE":  np.mean([m["MAE"]  for m in fold_metrics]),
                "mean_MAPE": np.mean([m["MAPE"] for m in fold_metrics]),
                "NRMSE":     mean_rmse / levels[col],
            })

    results_df = pd.DataFrame(rows)
    all_results_df[freq_name] = results_df

    print(f"\n=== {freq_name} ===")
    for metric in ["mean_RMSE", "mean_MAE", "mean_MAPE", "NRMSE"]:
        print(f"\n  {metric}:")
        print(
            results_df
            .pivot(index="variable", columns="horizon", values=metric)
            .round(4)
        )
        
        

with pd.ExcelWriter("out/arima/cv_results.xlsx", engine="openpyxl") as writer:
    for freq_name, results_df in all_results_df.items():
        results_df.to_excel(writer, sheet_name=freq_name, index=False)
          



plot_horizon = {'Monthly': 12, 'Weekly': 8, 'Daily_5day': 20, 'Daily_7day': 30, 'Quarterly': 4}
# how much history to show before the first fold forecast
plot_history = {'Monthly': 36, 'Weekly': 52, 'Daily_5day': 120, 'Daily_7day': 180, 'Quarterly': 8}

n_folds = 5

for freq_name in dataframes:
    df_orig    = dataframes[freq_name]
    cv_fore    = all_cv_forecasts[freq_name]
    results_df = all_results_df[freq_name]
    h_plot     = plot_horizon[freq_name]
    h_short    = key_horizons[freq_name][0]
    colors     = plt.cm.RdYlGn(np.linspace(0.2, 0.8, n_folds))

    cols   = df_orig.columns
    n_cols = 2
    n_rows = int(np.ceil(len(cols) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, n_rows * 4))
    axes = axes.flatten()

    for i, col in enumerate(cols):
        ax = axes[i]

        # find earliest forecast start across all folds for this col
        fold_entries = cv_fore[col][h_plot]
        if fold_entries:
            first_forecast_date = min(e['forecast'].index[0] for e in fold_entries)
            history_start = df_orig.index[
                max(0, df_orig.index.get_loc(first_forecast_date) - plot_history[freq_name])
            ]
        else:
            history_start = df_orig.index[-plot_history[freq_name]]

        ax.plot(df_orig[col].index, df_orig[col].values,
                color='black', linewidth=1, label='Actual', alpha=0.6, zorder=2)

        for fold_i, entry in enumerate(fold_entries):
            color = colors[fold_i % len(colors)]
            ax.plot(entry['forecast'], color=color, linewidth=1.5,
                    linestyle='--', alpha=0.9, label=f'Fold {fold_i+1}')
            ax.plot(entry['actual'],   color=color, linewidth=1.5, alpha=0.5)

        # zoom into the relevant window
        ax.set_xlim(history_start, df_orig.index[-1])

        nrmse_row = results_df[results_df['variable'] == col]
        nrmse_1   = nrmse_row[nrmse_row['horizon'] == h_short]['NRMSE'].values
        nrmse_h   = nrmse_row[nrmse_row['horizon'] == h_plot ]['NRMSE'].values
        title = f"{col}  |  NRMSE h={h_short}: {nrmse_1[0]:.3f}  h={h_plot}: {nrmse_h[0]:.3f}" \
                if len(nrmse_1) and len(nrmse_h) else col

        ax.set_title(title, fontsize=9)
        ax.legend(fontsize=7, ncol=3)
        ax.grid(alpha=0.3)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(f"CV Forecasts — {freq_name}", fontsize=12, y=1.01)
    plt.tight_layout()
    plt.savefig(f"out/arima/forecast_{freq_name.lower()}.pdf", bbox_inches='tight')
    plt.close()