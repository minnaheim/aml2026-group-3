# AML 2026 - Project 
## Forecasting Macro Variables based on Fed Speeches

By: Anna, Chris and Minna


## running the `holdout-validation`:

### specifying a target and including weights and biases and running on 'cuda':

`python src/holdout-validation/main.py --target CPI --wandb --device cuda`


### default targets, including weights and biases and running on 'cuda':

`python src/holdout-validation/main.py --wandb --device cuda`


### with embeddings (speeches) & rest

`python src/holdout-validation/main.py --embeddings --wandb --device cuda`
