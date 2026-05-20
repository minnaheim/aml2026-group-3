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
│   ├── holdout/
│   │   ├── metrics.csv                         # MAE/RMSE per model and target
│   │   ├── metrics_per_fold.csv                # per-fold metrics
│   │   ├── predictions_vs_actuals.png          # visual comparison across models
│   ├── cv/
│   │   ├── ar_orders/global.json               # CV-selected AR orders
│   │   └── arima_orders/global.json            # CV-selected ARIMA orders
│   └── tft/                                    # misc TFT experiment outputs
├── src/
│   ├── holdout-validation/
│   │   ├── a_embedding_manager.py      # manages FOMC speech embedding loading and alignment
│   │   ├── a_speech_attention.py       # speech attention analysis
│   │   ├── b_data_frame_builder.py     # loads macro series + optional speech embeddings; builds the panel dataframe
│   │   ├── c_benchmark_runner.py       # AR(1) and ARIMA runners (log-differencing, fitting, inverse transform)
│   │   ├── d_tft_runner.py             # TFT runner (PyTorch Forecasting)
│   │   ├── e_main.py                   # orchestrates all runners, logs to W&B, saves outputs
│   │   ├── f_run_ablation.py           # ablation study runner
│   │   ├── g_tune_hyperparams.py       # Bayesian HP tuning (Optuna/TPE), per-target
│   │   └── plot_experiments.py         # plotting utilities for experiment results
│   └── notebooks/                      # exploratory notebooks and scripts (not part of main pipeline)
```

---

## Running the pipeline

Default targets (`CPI`, `UNRATE`, `GDP`) on GPU with W&B logging:
```bash
python src/holdout-validation/e_main.py --wandb --device cuda
```

Single target:
```bash
python src/holdout-validation/e_main.py --target CPI --device cpu
```

With FOMC speech embeddings:
```bash
python src/holdout-validation/e_main.py --target CPI --embedding fomc-roberta --wandb --device cuda
```

Without FOMC speech embeddings:
```bash
python src/holdout-validation/e_main.py --wandb --device cuda
```

---

## Running the ablation

All embedding/aggregation combinations across all targets:
```bash
python src/holdout-validation/f_run_ablation.py --device cuda
```

Single target (quick test):
```bash
python src/holdout-validation/f_run_ablation.py --targets CPI --device cpu
```

Specific runs only:
```bash
python src/holdout-validation/f_run_ablation.py --runs macro_only fomc_roberta_mean --device cuda
```

---

## Hyperparameter tuning (Bayesian / Optuna)

<!-- Quick smoke test (3 trials):
```bash
python src/holdout-validation/g_tune_hyperparams.py --target CPI --n-trials 3 --device cuda
``` -->

**Observed runtime (Renku, CUDA):** ~5.5 min for 3 trials → ~1.8 min/trial. Full 50-trial sweep ≈ 1.5 h per target.

Best params are saved to `out/tuning/{target}/{run_tag}/best_params.json`. The study is checkpointed to `out/tuning/{target}/{run_tag}/optuna_study.pkl` and resumes automatically if re-run. `run_tag` is `macro_only` or `{embedding}_{aggregation}_{reduction}`.

---

### Macro-only (no embeddings) — ~4.5 h total
```bash
python src/holdout-validation/g_tune_hyperparams.py --target CPI --n-trials 40 --horizon 3 --device cuda --wandb
python src/holdout-validation/g_tune_hyperparams.py --target GDP --n-trials 40 --horizon 3 --device cuda --wandb
python src/holdout-validation/g_tune_hyperparams.py --target UNRATE --n-trials 40 --horizon 3 --device cuda --wandb
# fomc-roberta + mean aggregation
python src/holdout-validation/g_tune_hyperparams.py --target CPI   --n-trials 50 --embedding fomc-roberta --aggregation mean --reduction pca --device cuda --wandb
python src/holdout-validation/g_tune_hyperparams.py --target UNRATE --n-trials 50 --embedding fomc-roberta --aggregation mean --reduction pca --device cuda --wandb
python src/holdout-validation/g_tune_hyperparams.py --target GDP    --n-trials 50 --embedding fomc-roberta --aggregation mean --reduction pca --device cuda --wandb 
```


Once that is done, look at hyperparam, what stayed the same from all the runs, fix that!



<!-- ### fomc-roberta — ~4.5 h total (4 configs × 3 targets)
```bash
# mean aggregation
python src/holdout-validation/g_tune_hyperparams.py --target CPI   --n-trials 50 --embedding fomc-roberta --aggregation mean --reduction pca --device cuda
python src/holdout-validation/g_tune_hyperparams.py --target UNRATE --n-trials 50 --embedding fomc-roberta --aggregation mean --reduction pca --device cuda
python src/holdout-validation/g_tune_hyperparams.py --target GDP    --n-trials 50 --embedding fomc-roberta --aggregation mean --reduction pca --device cuda
``` -->

<!-- 
# # mean + factor analysis
# python src/holdout-validation/g_tune_hyperparams.py --target CPI   --n-trials 50 --embedding fomc-roberta --aggregation mean --reduction fa --device cuda
# python src/holdout-validation/g_tune_hyperparams.py --target UNRATE --n-trials 50 --embedding fomc-roberta --aggregation mean --reduction fa --device cuda
# python src/holdout-validation/g_tune_hyperparams.py --target GDP    --n-trials 50 --embedding fomc-roberta --aggregation mean --reduction fa --device cuda

# # decay aggregation
# python src/holdout-validation/g_tune_hyperparams.py --target CPI   --n-trials 50 --embedding fomc-roberta --aggregation decay --reduction pca --device cuda
# python src/holdout-validation/g_tune_hyperparams.py --target UNRATE --n-trials 50 --embedding fomc-roberta --aggregation decay --reduction pca --device cuda
# python src/holdout-validation/g_tune_hyperparams.py --target GDP    --n-trials 50 --embedding fomc-roberta --aggregation decay --reduction pca --device cuda

# # attention aggregation
# python src/holdout-validation/g_tune_hyperparams.py --target CPI   --n-trials 50 --embedding fomc-roberta --aggregation attention --reduction pca --device cuda
# python src/holdout-validation/g_tune_hyperparams.py --target UNRATE --n-trials 50 --embedding fomc-roberta --aggregation attention --reduction pca --device cuda
# python src/holdout-validation/g_tune_hyperparams.py --target GDP    --n-trials 50 --embedding fomc-roberta --aggregation attention --reduction pca --device cuda
# ```

# ---

# ### finbert — ~9 h total (2 configs × 3 targets)
# ```bash
# python src/holdout-validation/g_tune_hyperparams.py --target CPI   --n-trials 50 --embedding finbert --aggregation mean --reduction pca --device cuda
# python src/holdout-validation/g_tune_hyperparams.py --target UNRATE --n-trials 50 --embedding finbert --aggregation mean --reduction pca --device cuda
# python src/holdout-validation/g_tune_hyperparams.py --target GDP    --n-trials 50 --embedding finbert --aggregation mean --reduction pca --device cuda

# python src/holdout-validation/g_tune_hyperparams.py --target CPI   --n-trials 50 --embedding finbert --aggregation decay --reduction pca --device cuda
# python src/holdout-validation/g_tune_hyperparams.py --target UNRATE --n-trials 50 --embedding finbert --aggregation decay --reduction pca --device cuda
# python src/holdout-validation/g_tune_hyperparams.py --target GDP    --n-trials 50 --embedding finbert --aggregation decay --reduction pca --device cuda
# ``` -->