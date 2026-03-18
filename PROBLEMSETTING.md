# AML Proposal: Group 03 -- WORK IN PROGRESS

*What we would like to discuss is in the last section. We have two possible ideas which we both discuss below.*

## Problem Setting 

**Historical Context and Motivation**

The *Federal Reserve Act* of 1913 established the Federal Reserve (Fed) as the central bank of the United States. After undergoing several institutional changes, the Fed's structure and that of its policymaking organ, the Federal Open Market Committee (FOMC), have remained unchanged since 1943. One notable feature of the Fed is its statutory obligation to publicly disclose any communications by its officials. This means that all speeches (e.g., at conferences) must be made available to the public either by inviting journalists or through live streaming. These transparency requirements closely align with more recent policymaking by central bankers. Since the 1990s, the Fed has aimed to increase its transparency, for example through the immediate post-meeting policy statements, for two reasons. First, central bankers aim to strengthen the legitimacy and independence of monetary policy. Second, communication serves as a critical policy tool to signal the path of future monetary policy and guide market expectations (*Forward Guidance*).

<!-- ich glaube diese info ist nicht soo relevant, oder?
The FOMC consists of 19 members: seven members of the Board of Governors and 12 presidents of regional Fed branches. Only 12 of all members have voting rights. Members of the Board and the president of the New York Fed can vote every year while the voting rights of the other regional presidents follow an exogenous rotation system.  -->

The recent literature has frequently discussed whether the Fed possesses **superior information**. This can refer either to information that is not generally available to the public or to a better processing of the publicly available information. We examine this question from a forecasting perspective. Can we extract information from Fed speeches that goes beyond information from publicly available data and use it to improve forecasts of macroeconomic conditions? If so, this would suggest that the Fed does indeed possess superior information. If we were to find such an effect, it would not fall under the Lucas critique: instead, it would successfully show that the Fed uses Forward Guidance to align market expectations with its monetary policy goals.

**Research Question: Do Fed Speeches contain information useful to forecast macroeconomic indicators?**

## Idea 1: Using SSMs

**Formal Setting**:

Is the current information set that we have the same as the (communicated) information set of the Federal Reserve (Fed)? Meaning, does the Fed communicate more than what is already publicly available? If yes, including their communication in forecasting can improve the forecasting of Macro Variables.

1. **Input**
  - Macroeconomic Data: CPI, unemployment rate, monthly exchange rates against Yen, GBP, EUR (these are variables typically used in the literature)
  - Fed Speeches 
    - Embedded via [FOMC-RoBERTa](https://huggingface.co/gtfintechlab/FOMC-RoBERTa), [FinBERT](https://arxiv.org/abs/1908.10063) or [Cental Bank RoBERTa](https://github.com/Moritz-Pfeifer/CentralBankRoBERTa)
    - TBD: BERT embeddings only use fixed token size as input (i.e. 512) -> so after cleaning, we just take the first 512 tokens?
    - TBD: Post-embeddings, do we turn these embedded texts at time points into monthly averaged index (?)
2. **Output**
  - Forecasts of macroeconomic variables (CPI, ...)
  - SSM hidden state as an index for another model?
3. **Model**

  - State Space Model:
    - [Dynamic Factor SSM](https://cran.r-project.org/web/packages/dfms/vignettes/dynamic_factor_models.pdf) or [see this book](https://www.princeton.edu/~mwatson/papers/Stock_Watson_HOM_Vol2.pdf) which is very econonomics oriented
    - Time varying parameter SSM (very economics oriented)
  <!-- - Bayesian SSM^[See KOF PhD thesis on Bayesian SSMs: https://www.research-collection.ethz.ch/server/api/core/bitstreams/da15bb15-9e0e-43bf-b329-df754eb81510/content] -->


## Idea 2: Using Temporal Fusion Transformer (TFT)

Is the current information set that we have the same as the (communicated) information set of the Federal Reserve (Fed)? Meaning, does the Fed communicate more than what is already publicly available? If yes, including their communication in forecasting can improve the forecasting of Macro Variables.

1. **Input**
  - Macroeconomic Data: CPI, unemployment rate, monthly exchange rates against Yen, GBP, EUR (these are variables typically used in the literature)
  - Fed Speeches 
    - Alignment to *next* inflation (...) release data; no look ahead to ensure no data leakage
    - Extract embeddings from [FOMC-RoBERTa](https://huggingface.co/gtfintechlab/FOMC-RoBERTa), [FinBERT](https://arxiv.org/abs/1908.10063) or [Cental Bank RoBERTa](https://github.com/Moritz-Pfeifer/CentralBankRoBERTa)
    - Ideally fine-tune the model used for embedding first on a proxy task (e.g., predict fed funds direction at next meeting, so hike/hold/cut)
    - Then: Dimensionality Reduction
  - Static features (speaker characteristics) from feature Engineering: we have access to a wide variety of speaker and institution features (Lustenberger, Rossi and Zeitz, Central Bank Communication: New Data and Stylized Facts From a Century of Fed Speeches, forthcoming as SNB Working Paper, 2026.)
2. **Output**
  - Forecasts of macroeconomic variables (CPI, ...)
3. **Model**

  - [Forecasting using Transformers](https://www.sciencedirect.com/science/article/pii/S0169207021000637)
  - [Temporal Fusion Transformers for Interpretable Multi-horizon Time Series Forecasting](https://arxiv.org/abs/1912.09363)



## Evaluation Protocol
- Cross-Validation (n-folds dep. on available data, *tbd since we use temporal data*)
- Error Metrics:
  - Mean RMSE (over all folds)
  - Mean Average Error
  - Mean Average Percentage Error
- Use of Vintages of Macro Variables^[We assume Fed Speeches are unrevised]
- Horizon
  - for *Monthly* Variables (CPI, unemployment rate, exchange rates): 1,6,12
  - for *Quarterly* Variables (GDP): 1,4
 


## Hyperparameter Tuning
- tbd. based on SSM/TFT use

## Machine Learning Benchmark
- tbd.

- Model itself without text embeddings (do we improve forecasts or do we introduce noise?)

## Statistical Benchmark
- Stationarity Checks 
- AR(1) or AR(2)


## Our Questions / To Discuss

In class, we saw that we will discuss SSMs such as Mamba. Dynamic Factor SSM are particularly useful for economic datasets as it accounts for the i) short time dimension (total length and data frequency) and ii) potentially many different time series. However, we are not certain if this fits the exercise requirements. 

Alternatively, TFTs would likely be particularly useful for our research question. However, we are not certain if this fits the exercise requirements.

<!-- 
### Questions to ask 

- Granger Causality Tests (?)
- FinBert takes fixed number of tokens as input → how do do this?
- any specific recommendations for the SSM? Mamba, e.g.? -->


#### Sources 

- https://arxiv.org/abs/2506.22763
- https://www.alexandria.unisg.ch/server/api/core/bitstreams/1d94cc0d-30b9-4d0d-9131-8e8c20c46837/content (finetuning FinBERT to FOMC minutes)
