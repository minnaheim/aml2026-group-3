# AML Proposal: Group 03


## Problem Setting 

**(Historical) Context:**

- Period of "Great Inflation" (intransparency)
- legally required to publish transscripts of Fed Meetings 
- transparency important for economic signalling, forward guidance
- Lucas Critique^[This means that the Fed does not adjust its behavior to our insights which might be plausible in light of the forward guidance goals.]

**Formal Setting**:

Is the current information set that we have the same as the (communicated) information set of the Federal Reserve (Fed)? Meaning, does the Fed communicate more than what is already publicly available? If yes, including their communication in forecasting can improve the forecasting of Macro Variables.

1. **Input**
  - Macro Data (i.e. CPI, unemployment rate, monthly exchange rates)
  - Fed Speeches (embedded via FinBERT or [Cental Bank RoBERTa](https://github.com/Moritz-Pfeifer/CentralBankRoBERTa) and then as Index?)
2. **Output**
  - predicted macro variables 
  - SSM hidden state as an index for another model?
3. **Model**
  - State Space Model 
  - Comp. Efficient MAMBA?
  - Bayesian SSM^[See KOF PhD thesis on Bayesian SSMs: https://www.research-collection.ethz.ch/server/api/core/bitstreams/da15bb15-9e0e-43bf-b329-df754eb81510/content]
  - Dynamic Factor SSM (very econ oriented though)

## Evaluation Protocol
- Cross-Validation (n-folds dep. on available data)
- Error Metrics:
  - Mean RMSE (over all folds)
  - Mean Average Error
  - Mean Average Percentage Error
- Use of Vintages of Macro Variables^[We assume Fed Speeches are unrevised]
- Horizon
  - for *Monthly* Variables (CPI, unemployment rate, exchange rates): 1,6,12
  - for *Quarterly* Variables (GDP): 1,4 

## Hyperparameter Tuning
- tbd. based on SSM use

## Machine Learning Benchmark
- tbd. 

## Statistical Benchmark
- Stationarity Checks 
- AR(1) or AR(2)


### Questions to ask 

- Granger Causality Tests (?)
- FinBert takes fixed number of tokens as input → how do do this?
- any specific recommendations for the SSM? Mamba, e.g.?


#### Sources 

- https://arxiv.org/abs/2506.22763