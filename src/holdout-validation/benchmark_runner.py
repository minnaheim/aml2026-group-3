import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller

# mirrors AR-benchmark.py: CPI/PAYEMS/INDPRO/GDP are log-differenced before fitting
# UNRATE is mean-reverting in practice and is kept in levels
LOG_DIFF_TARGETS = {"CPI", "PAYEMS", "INDPRO", "GDP"} # log, then diff?


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

# this is taken care of in main, no?
    # @staticmethod
    # def compute_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict:
    #     errors = actual - predicted
    #     mae  = np.mean(np.abs(errors))
    #     rmse = np.sqrt(np.mean(errors ** 2))
    #     mask = actual != 0
    #     mape = np.mean(np.abs(errors[mask] / actual[mask])) * 100
    #     return {"MAE": mae, "RMSE": rmse, "MAPE": mape}
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


# try it out here! 

# from data_frame_builder import DataFrameBuilder 

# path = "/Users/minna/Code/FS26/AML/aml2026-group-3"
# dfb = DataFrameBuilder(path)
# df = dfb.process_data()

# splits, holdout = dfb.generate_split(df)

# for s in splits:
#     tr, te = s["train"], s["test"]
#     print(f"Fold {s['fold']}: train [{tr['date'].min().date()} – {tr['date'].max().date()}] ({len(tr)} rows) | "
#           f"test [{te['date'].min().date()} – {te['date'].max().date()}] ({len(te)} rows)")

# arr = ARRunner(dfb)
# result = arr.run(splits, target="CPI", fold = 0)
# print(result)

# # plot the result 
# import matplotlib.pyplot as plt
# from pathlib import Path

# out_dir = Path(path) / "out" / "ar1"
# out_dir.mkdir(parents=True, exist_ok=True)

# fig, ax = plt.subplots(figsize=(12, 4))
# ax.plot(result["date"], result["actual"],    label="Actual",    color="black",  linewidth=1.5)
# ax.plot(result["date"], result["predicted"], label="Predicted", color="crimson", linewidth=1.5, linestyle="--")
# ax.set_title(f"AR(1) — {result['target'].iloc[0]}: Predicted vs Actual")
# ax.set_xlabel("Date")
# ax.legend()
# ax.grid(alpha=0.3)
# plt.tight_layout()
# plt.savefig(out_dir / f"ar1_{result['target'].iloc[0].lower()}.png", bbox_inches="tight")
# plt.show()

# conclusion: AR(1) performs badly, especially over such a long horizon. with shorter horizon, aka more folds it would be much better. i guess this helps our tft...