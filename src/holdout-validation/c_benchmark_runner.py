import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller
from pmdarima import auto_arima

from pathlib import Path
import json

# mirrors AR-benchmark.py: CPI/PAYEMS/INDPRO/GDP are log-differenced before fitting
# UNRATE is mean-reverting in practice and is kept in levels

# ---------------------------------------------------------------------------
# AR Runner
# ---------------------------------------------------------------------------
class ARRunner:
    def __init__(self, dfb, cache_dir: Path | None = None, max_prediction_length: int = 12):
        self.dfb = dfb
        self.max_prediction_length = max_prediction_length
        self._cache_dir = cache_dir or (Path(dfb.path) / "out" / "cv" / "ar_orders")
        self._cache_dir.mkdir(parents=True, exist_ok=True)

    def _cache_path(self, key: str) -> Path:
        return self._cache_dir / f"{key}.json"

    def _load_orders(self, key: str) -> dict:
        p = self._cache_path(key)
        return json.load(open(p)) if p.exists() else {}

    def _save_orders(self, key: str, orders: dict) -> None:
        json.dump(orders, open(self._cache_path(key), "w"), indent=2)

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
                # max AR(p=3)
                start_p=0, max_p=3, start_q=0, max_q=0,
                start_P=0, max_P=0, start_Q=0, max_Q=0, # no seasonal order fit
                m=m,
                seasonal=True,
                information_criterion="aic",
                error_action="ignore",
                suppress_warnings=True,
                stepwise=True,
            )
            order, seasonal_order = fit.order, fit.seasonal_order
        except Exception as exc:
            print(f"auto_arima failed for {cache_key}: {exc}. Falling back to (1,0,0).")
            order, seasonal_order = (1, 0, 0), (0, 0, 0, 0)

        cached_orders[cache_key] = {
            "order": list(order),
            "seasonal_order": list(seasonal_order),
            "selection_end": str(selection_end),
        }
        return order, seasonal_order

    # main run
    def run(self, splits, target: str = "CPI", fold: int = 0) -> tuple:
        train = self.dfb.get_data(splits, train=True,  model="AR", target=target, fold=fold)

        freq   = "QS" if target in self.dfb.QUARTERLY_TARGETS else "MS"
        stationary = train.set_index("date")[target]

        # use end of training data as selection window for order search
        selection_end = stationary.index.max()
        min_required  = 24 if freq == "MS" else 8
        cached_orders = self._load_orders("global")
        cache_key     = f"{target}_{freq}_fold{fold}"  # fold-specific to avoid cross-fold pollution

        if len(stationary.dropna()) >= min_required:
            order, seasonal_order = self._find_order(
                stationary, freq, cache_key, cached_orders, selection_end
            )
        else:
            print(f"Insufficient data for {target}, falling back to (1,0,0).")
            order, seasonal_order = (1, 0, 0), (0, 0, 0, 0)

        self._save_orders("global", cached_orders)

        model = SARIMAX(
            stationary.asfreq(freq),
            order=order,
            seasonal_order=seasonal_order,
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        self._fit            = model.fit(maxiter=250, disp=False)


        print(f"[AR | {target}] order={order}, seasonal={seasonal_order}, ")

        return order, seasonal_order
    
    def predict(self, splits, target: str = "CPI", fold: int = 0, step: int | None = None) -> pd.DataFrame:
        step = step if step is not None else self.max_prediction_length
        train = self.dfb.get_data(splits, train=True,  model="AR", target=target, fold=fold)
        test  = self.dfb.get_data(splits, train=False, model="AR", target=target, fold=fold)

        freq     = "QS" if target in self.dfb.QUARTERLY_TARGETS else "MS"
        test_len = len(test)
        windows: list[pd.DataFrame] = []
        pred_context = train.copy()  # grows with own predictions, never actual test obs

        for start in range(0, test_len, step):
            end         = min(start + step, test_len)
            window_size = end - start
            test_window = test.iloc[start:end]

            # context = train + own predictions so far (no actual test obs — no pollution)
            # drop duplicate dates at the train/test boundary (can happen when split doesn't fall on a quarter boundary)
            context_raw    = pred_context.drop_duplicates(subset="date", keep="last")
            context_series = context_raw.set_index("date")[target]

            # expanding context mirrors TFT: same fixed params (from run()), but encoder sees more predicted data
            # .apply() keeps the fitted params from run(), no refit
            ctx_result   = self._fit.apply(context_series.asfreq(freq))
            fc_obj       = ctx_result.get_forecast(steps=window_size)
            forecast     = fc_obj.predicted_mean.values
            # 80% CI = 10th–90th quantile, matching e_main.py pinball loss at q=0.1/0.9
            ci           = fc_obj.conf_int(alpha=0.2).values

            # append own predictions (not actuals) to context for next window
            pred_rows    = pd.DataFrame({"date": test_window["date"].values, target: forecast})
            pred_context = pd.concat([pred_context, pred_rows], ignore_index=True)

            windows.append(pd.DataFrame({
                "date":      test_window["date"].values,
                "actual":    test_window[target].values.astype(float),
                "predicted": forecast.astype(float),
                "pred_lo":   ci[:, 0].astype(float),
                "pred_hi":   ci[:, 1].astype(float),
                "target":    target,
            }))

        return pd.concat(windows, ignore_index=True)


# ---------------------------------------------------------------------------
# ARIMA Runner
# ---------------------------------------------------------------------------

class ARIMARunner:
    """5-fold CV ARIMA with auto_arima order search (cached per fold/variable)."""

    def __init__(self, dfb, cache_dir: Path | None = None, max_prediction_length: int = 12):
        self.dfb = dfb
        self.max_prediction_length = max_prediction_length
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
    def run(self, splits, target: str = "CPI", fold: int = 0) -> tuple:
        train = self.dfb.get_data(splits, train=True,  model="ARIMA", target=target, fold=fold)

        freq   = "QS" if target in self.dfb.QUARTERLY_TARGETS else "MS"
        series = train.set_index("date")[target]
        stationary = series.dropna()

        # use end of training data as selection window for order search
        selection_end = series.index.max()
        min_required  = 24 if freq == "MS" else 8
        cached_orders = self._load_orders("global")
        cache_key     = f"{target}_{freq}_fold{fold}"  # fold-specific to avoid cross-fold pollution

        if len(stationary.dropna()) >= min_required:
            order, seasonal_order = self._find_order(
                stationary, freq, cache_key, cached_orders, selection_end
            )
        else:
            print(f"Insufficient data for {target}, falling back to (1,0,1).")
            order, seasonal_order = (1, 0, 1), (0, 0, 0, 0)

        self._save_orders("global", cached_orders)

        model = SARIMAX(
            stationary.asfreq(freq),
            order=order,
            seasonal_order=seasonal_order,
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        self._fit            = model.fit(maxiter=250, disp=False)

        print(f"[ARIMA | {target}] order={order}, seasonal={seasonal_order}, "
              f"n_train={len(stationary)}")

        return order, seasonal_order

    def predict(self, splits, target: str = "CPI", fold: int = 0, step: int | None = None) -> pd.DataFrame:
        step = step if step is not None else self.max_prediction_length
        train = self.dfb.get_data(splits, train=True,  model="ARIMA", target=target, fold=fold)
        test  = self.dfb.get_data(splits, train=False, model="ARIMA", target=target, fold=fold)

        freq     = "QS" if target in self.dfb.QUARTERLY_TARGETS else "MS"
        test_len = len(test)
        windows: list[pd.DataFrame] = []
        pred_context = train.copy()  # grows with own predictions, never actual test obs

        for start in range(0, test_len, step):
            end         = min(start + step, test_len)
            window_size = end - start
            test_window = test.iloc[start:end]

            # context = train + own predictions so far (no actual test obs — no pollution)
            # drop duplicate dates at the train/test boundary (can happen when split doesn't fall on a quarter boundary)
            context_raw    = pred_context.drop_duplicates(subset="date", keep="last")
            context_series = context_raw.set_index("date")[target]

            # expanding context mirrors TFT: same fixed params (from run()), but model sees more predicted data
            # .apply() keeps the fitted params from run(), no refit
            ctx_result   = self._fit.apply(context_series.asfreq(freq))
            fc_obj       = ctx_result.get_forecast(steps=window_size)
            forecast     = fc_obj.predicted_mean.values
            # 80% CI = 10th–90th quantile, matching e_main.py pinball loss at q=0.1/0.9
            ci           = fc_obj.conf_int(alpha=0.2).values

            # append own predictions (not actuals) to context for next window
            pred_rows    = pd.DataFrame({"date": test_window["date"].values, target: forecast})
            pred_context = pd.concat([pred_context, pred_rows], ignore_index=True)

            windows.append(pd.DataFrame({
                "date":      test_window["date"].values,
                "actual":    test_window[target].values.astype(float),
                "predicted": forecast.astype(float),
                "pred_lo":   ci[:, 0].astype(float),
                "pred_hi":   ci[:, 1].astype(float),
                "target":    target,
            }))

        return pd.concat(windows, ignore_index=True)
