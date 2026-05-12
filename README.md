# AML 2026 - Project
## Forecasting Macro Variables based on Fed Speeches

By: Anna, Chris and Minna

---

## Overview

We forecast macroeconomic variables (Main targets: CPI, UNRATE, GDP) over a 12-month holdout window. Three models are compared on identical information sets and forecast horizons: **AR(1)**, **ARIMA**, (our statistical benchmarks) and **TFT** (our ML benchmark). FOMC speech embeddings can optionally be included as additional features (main model).

For model choices, data sources, and methodology see `PREPARE_PRESENTATION.md`.

Below you will find a project overview of the most important files.

```md
├── data/
├── out/
│   └── holdout/
│       ├── metrics.csv                 # MAE/RMSE per model and target
│       ├── predictions_vs_actuals.png  # visual comparison across models
│       └── *_predictions.csv          # per-target prediction files
├── src/
│   └── holdout-validation/
│       ├── data_frame_builder.py       # loads macro series + optional speech embeddings; builds the panel dataframe
│       ├── benchmark_runner.py         # AR(1) and ARIMA runners (log-differencing, fitting, inverse transform)
│       ├── tft_runner.py               # TFT runner (PyTorch Forecasting)
│       └── main.py                     # orchestrates all runners, logs to W&B, saves outputs
```

---

## Running the pipeline

Default targets (`CPI`, `UNRATE`, `GDP`) on GPU with W&B logging:
```bash
python src/holdout-validation/main.py --wandb --device cuda
```

Single target:
```bash
python src/holdout-validation/main.py --wandb --device cpu
```

With FOMC speech embeddings:
```bash
python src/holdout-validation/main.py --embeddings fomc-roberta --wandb --device cuda
```

With FOMC speech embeddings:
```bash
python src/holdout-validation/main.py --wandb --device cuda
```
