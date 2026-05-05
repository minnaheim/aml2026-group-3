import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller
from pmdarima import auto_arima

from pathlib import Path
import json

# mirrors AR-benchmark.py: CPI/PAYEMS/INDPRO/GDP are log-differenced before fitting
# UNRATE is mean-reverting in practice and is kept in levels
LOG_DIFF_TARGETS = {"CPI", "PAYEMS", "INDPRO", "GDP"} # log, then diff?


# ---------------------------------------------------------------------------
# AR Runner
# ---------------------------------------------------------------------------
class ARRunner:
    def __init__(self, dfb):
        self.dfb = dfb

    # only needed internally
    def _transform(self, series: pd.Series, target: str):
        """Apply the same transform as AR-benchmark.py."""
        if target in LOG_DIFF_TARGETS:
            # test if we really need both
            log = np.log(series)
            log_diff = log.diff().dropna()
            print(f"just log adf test: {adfuller(log)[1]}")
            # claude says the reason its stationary here but not when differentiating is due to misspecification of adfuller, regression='c' should be 'ct' tho.
            print(f"log & diff adf test: {adfuller(log_diff)[1]}")
            # here, we have structural breaks in CPI, since such long time frame.
            return np.log(series).diff().dropna(), "log_diff"
        return series.copy(), "levels"  # UNRATE: already stationary in levels

    def _invert(self, forecast: np.ndarray, last_val: float, transform: str) -> np.ndarray:
        """Invert transform to get forecasts back in original units."""
        if transform == "log_diff":
            # mirrors invert_transform(..., is_log=True) in AR-benchmark.py
            return np.exp(np.log(last_val) + np.cumsum(forecast))
        return forecast  # levels: SARIMAX forecast is already in original units


    def _fetch_data(self, splits, target: str = "CPI", fold: int = 0):
        train = self.dfb.get_data(splits, train=True,  model="AR", target=target, fold=fold)
        test  = self.dfb.get_data(splits, train=False, model="AR", target=target, fold=fold)
        return train, test


    def run(self, splits, target: str = "CPI", fold: int = 0) -> pd.DataFrame:
        train, test = self._fetch_data(splits, target, fold)

        series     = train.set_index("date")[target]
        freq       = "QS" if target in self.dfb.QUARTERLY_TARGETS else "MS"

        # TODO: is it a problem that we transform both separately??
        # transform both to stationarity if needed
        stationary, transform = self._transform(series, target)

        print(f"[AR | {target}] transform={transform}, n_train={len(stationary)}, n_test={len(test)}")

        # fixed AR(1) — same spec as AR-benchmark.py: SARIMAX(1,0,0)
        model = SARIMAX(
            stationary.asfreq(freq),
            # no choice here...
            order=(1, 0, 0),
            seasonal_order=(0, 0, 0, 0),
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        result   = model.fit(maxiter=1000, disp=False)
        forecast_raw = result.get_forecast(steps=len(test)).predicted_mean.values

        forecast = self._invert(forecast_raw, series.iloc[-1], transform)

        return pd.DataFrame({
            "date":      test["date"].values,
            "actual":    test[target].values,
            "predicted": forecast,
            "target":    target,
        })
        
        
# ---------------------------------------------------------------------------
# ARIMA Runner
# ---------------------------------------------------------------------------

class ARIMARunner:
    """5-fold CV ARIMA with auto_arima order search (cached per fold/variable)."""

    def __init__(self, dfb, cache_dir: Path | None = None):
        self.dfb = dfb
        # saving orders & their performance
        self._cache_dir = cache_dir or (Path(dfb.path) / "out" / "cv" / "arima_orders")
        self._cache_dir.mkdir(parents=True, exist_ok=True)

    # cache helper
    def _cache_path(self, key: str) -> Path:
        return self._cache_dir / f"{key}.json"

    # directly load data
    def _load_orders(self, key: str) -> dict:
        p = self._cache_path(key)
        return json.load(open(p)) if p.exists() else {}

    def _save_orders(self, key: str, orders: dict) -> None:
        json.dump(orders, open(self._cache_path(key), "w"), indent=2)
        
    # transform helpers for differenced and log variables
    def _transform(self, series: pd.Series, target: str) -> tuple[pd.Series, str]:
        if target in LOG_DIFF_TARGETS:
            return np.log(series).diff().dropna(), "log_diff"
        return series.copy(), "levels"

    def _invert(self, forecast: np.ndarray, last_val: float, transform: str) -> np.ndarray:
        if transform == "log_diff":
            return np.exp(np.log(last_val) + np.cumsum(forecast))
        return forecast

    # order selection
    def _find_order(
        self,
        train_series: pd.Series,
        freq: str,
        cache_key: str,
        cached_orders: dict,
        selection_end,
    ) -> tuple[tuple, tuple]:
        if cache_key in cached_orders:
            entry = cached_orders[cache_key]
            if entry.get("selection_end") == str(selection_end):
                return tuple(entry["order"]), tuple(entry["seasonal_order"])
            else:
                print(f"Cache stale for {cache_key}, refitting.")

        m = 12 if freq == "MS" else 4
        try:
            fit = auto_arima(
                train_series.dropna(),
                d=0, D=0,
                start_p=0, max_p=3, start_q=0, max_q=3,
                start_P=0, max_P=2, start_Q=0, max_Q=2,
                m=m,
                seasonal=True,
                information_criterion="aic",
                error_action="ignore",
                suppress_warnings=True,
                stepwise=True,
            )
            order, seasonal_order = fit.order, fit.seasonal_order
        except Exception as exc:
            print(f"auto_arima failed for {cache_key}: {exc}. Falling back to (1,0,1).")
            order, seasonal_order = (1, 0, 1), (0, 0, 0, 0)

        cached_orders[cache_key] = {
            "order": list(order),
            "seasonal_order": list(seasonal_order),
            "selection_end": str(selection_end),
        }
        return order, seasonal_order

    # main run
    def run(self, splits, target: str = "CPI", fold: int = 0) -> pd.DataFrame:
        train = self.dfb.get_data(splits, train=True,  model="ARIMA", target=target, fold=fold)
        test  = self.dfb.get_data(splits, train=False, model="ARIMA", target=target, fold=fold)

        freq   = "QS" if target in self.dfb.QUARTERLY_TARGETS else "MS"
        series = train.set_index("date")[target]

        stationary, transform = self._transform(series, target)

        # use end of training data as selection window for order search
        selection_end = series.index.max()
        min_required  = 24 if freq == "MS" else 8
        cached_orders = self._load_orders("global")
        cache_key     = f"{target}_{freq}"

        if len(stationary.dropna()) >= min_required:
            order, seasonal_order = self._find_order(
                stationary, freq, cache_key, cached_orders, selection_end
            )
        else:
            print(f"Insufficient data for {target}, falling back to (1,0,1).")
            order, seasonal_order = (1, 0, 1), (0, 0, 0, 0)

        self._save_orders("global", cached_orders)

        print(f"[ARIMA | {target}] order={order}, seasonal={seasonal_order}, "
              f"transform={transform}, n_train={len(stationary)}, n_test={len(test)}")

        model = SARIMAX(
            stationary.asfreq(freq),
            order=order,
            seasonal_order=seasonal_order,
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        res          = model.fit(maxiter=250, disp=False)
        forecast_raw = res.get_forecast(steps=len(test)).predicted_mean.values
        forecast     = self._invert(forecast_raw, series.iloc[-1], transform)

        return pd.DataFrame({
            "date":      test["date"].values,
            "actual":    test[target].values,
            "predicted": forecast,
            "target":    target,
        })

# ---------------------------------------------------------------------------
# try it out here! 
# ---------------------------------------------------------------------------


# from data_frame_builder import DataFrameBuilder 

# path = "/Users/minna/Code/FS26/AML/aml2026-group-3"
# path = r"C:\Users\annaz\OneDrive\Dokumente\Studium\UZH_Master\2026FS\Advanced Machine Learning\Practical_Assignment\aml2026-group-3"
# dfb = DataFrameBuilder(path)
# df = dfb.process_data()

# splits, holdout = dfb.generate_split(df)

# for s in splits:
#     tr, te = s["train"], s["test"]
#     print(f"Fold {s['fold']}: train [{tr['date'].min().date()} – {tr['date'].max().date()}] ({len(tr)} rows) | "
#           f"test [{te['date'].min().date()} – {te['date'].max().date()}] ({len(te)} rows)")

# arr = ARRunner(dfb)
# result = arr.run(splits, target="CPI", fold = 0)
# arimar = ARIMARunner(dfb)
# result = arimar.run(splits, target="CPI", fold = 0)
# print(result)

# # # plot the result 
# import matplotlib.pyplot as plt
# from pathlib import Path

# out_dir = Path(path) / "out" / "ar1"
# out_dir = Path(path) / "out" / "arima"
# out_dir.mkdir(parents=True, exist_ok=True)

# fig, ax = plt.subplots(figsize=(12, 4))
# ax.plot(result["date"], result["actual"],    label="Actual",    color="black",  linewidth=1.5)
# ax.plot(result["date"], result["predicted"], label="Predicted", color="crimson", linewidth=1.5, linestyle="--")
# ax.set_title(f"AR(1) — {result['target'].iloc[0]}: Predicted vs Actual")
# ax.set_title(f"ARIMA — {result['target'].iloc[0]}: Predicted vs Actual")
# ax.set_xlabel("Date")
# ax.legend()
# ax.grid(alpha=0.3)
# plt.tight_layout()
# plt.savefig(out_dir / f"ar1_{result['target'].iloc[0].lower()}.png", bbox_inches="tight")
# plt.savefig(out_dir / f"arima_{result['target'].iloc[0].lower()}.png", bbox_inches="tight")

# conclusion: AR(1) performs badly, especially over such a long horizon. with shorter horizon, aka more folds it would be much better. i guess this helps our tft...
# conclusion: ARIMA also performs quite terribly