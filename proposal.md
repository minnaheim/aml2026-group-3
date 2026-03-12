# AML Proposal: Group 03


## Problem Setting 

**Historical Context and Motivation**

The *Federal Reserve Act* of 1913 established the Federal Reserve (Fed) as the central bank of the United States. After undergoing several institutional changes, the Fed's structure and that of its policymaking organ, the Federal Open Market Committee (FOMC), have remained unchanged since 1943. The FOMC consists of 19 members: seven members of the Board of Governors and 12 presidents of regional Fed branches. Only 12 of all members have voting rights. Members of the Board and the president of the New York Fed can vote every year while the voting rights of the other regional presidents follow an exogenous rotation system. 

One notable feature of the Fed is its statutory obligation to publicly disclose any communications by its officials. This means that all speeches (e.g., at conferences) must be made available to the public either by inviting journalists or through live streaming. These transparency requirements closely align with more recent policymaking by central bankers. Since the 1990s, the Fed has aimed to increase its transparency, for example through the immediate post-meeting policy statements, for two reasons. First, central bankers aim to strengthen the legitimacy and independence of monetary policy. Second, communication serves as a critical policy tool to signal the path of future monetary policy and guide market expectations (*Forward Guidance*).

The recent literature has frequently discussed whether the Fed possesses **superior information**. This can refer either to information that is not generally available to the public or to a better processing of the publicly available information. We examine this question from a forecasting perspective. Can we extract information from Fed speeches that goes beyond information from publicly available data and use it to improve forecasts of macroeconomic conditions? If so, this would suggest that the Fed does indeed possess superior information. If we were to find such an effect, it would not fall under the Lucas critique: instead, it would successfully show that the Fed uses Forward Guidance to align market expectations with its monetary policy goals.

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
