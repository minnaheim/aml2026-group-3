# AML Proposal: Group 03 -- WORK IN PROGRESS

## Problem Setting 

**Historical Context and Motivation**

The *Federal Reserve Act* of 1913 established the Federal Reserve (Fed) as the central bank of the United States. After undergoing several institutional changes, the Fed's structure and that of its policymaking organ, the Federal Open Market Committee (FOMC), have remained unchanged since 1943. One notable feature of the Fed is its statutory obligation to publicly disclose any communications by its officials. This means that all speeches (e.g., at conferences) must be made available to the public either by inviting journalists or through live streaming. These transparency requirements closely align with more recent policymaking by central bankers. Since the 1990s, the Fed has aimed to increase its transparency, for example through the immediate post-meeting policy statements, for two reasons. First, central bankers aim to strengthen the legitimacy and independence of monetary policy. Second, communication serves as a critical policy tool to signal the path of future monetary policy and guide market expectations (*Forward Guidance*).

The recent literature has frequently discussed whether the Fed possesses **superior information**. This can refer either to information that is not generally available to the public or to a better processing of the publicly available information. We examine this question from a forecasting perspective. Can we extract information from Fed speeches that goes beyond information from publicly available data and use it to improve forecasts of macroeconomic conditions? If so, this would suggest that the Fed does indeed possess superior information. If we were to find such an effect, it would not fall under the Lucas critique: instead, it would successfully show that the Fed uses Forward Guidance to align market expectations with its monetary policy goals.

**Research Question: Do Fed Speeches contain information useful to forecast macroeconomic indicators?**

## Formal Setting

We define $\boldsymbol{X}\in \mathbb{R}^{T\times n}$ as a matrix of $n$ variables observed over $T$ time periods where $\boldsymbol{X}^{(t)}\in \mathbb{R}^{t\times n}$ is a truncation of $\boldsymbol{X}$ up to $t\leq T$.
Let there be an information set defined as 

$$\mathcal{I}_t^{\text{public}} = \\{ \boldsymbol{X}^{(t)} \in \mathbb{R}^{t\times n}: \boldsymbol{X} \text{ observable to the public at } t \\}$$

which reflects the information the general public such as financial markets has available at time $t$. Define the information set by the Fed as

$$\mathcal{I}_t^{\text{fed}} = \mathcal{I}_t^{\text{public}} \cup \mathcal{I}_t^{\text{fed, excl}}$$

 where $\mathcal{I}\_t^{\text{fed, excl}}$ measures any private ("superior") information the Fed possesses. If $\mathcal{I}\_t^{\text{fed, excl}} = \emptyset$, then the Fed does not possess superior information and $\mathcal{I}\_t^{\text{fed}} = \mathcal{I}\_t^{\text{public}}$. The Fed can communicate with the public through some speech $s_{kt}$ which denotes the $k$-th speech at time $t$. Then, the set of all speeches given up to time $T$ is defined by

$$ \mathcal{S} = \\{s_{kt}: k = 1, ..., K_t; t = 1, ..., T\\}.$$

The Fed uses speeches $\mathcal{S}$ as a **forward guidance** mechanism to influence $\mathcal{I}\_t^{\text{public}}$. If $\mathcal{I}\_t^{\text{fed, excl}} \neq \emptyset$, the Fed communicates its superior information through $s_{kt}$.
However, Fed communication is complex and not perfectly interpretable. Define a *comprehension operator* $\mathcal{C}(\cdot)$ such that the public extracts from a speech $s_{kt}$:

$$\mathcal{C}(s_{kt}) \subseteq \mathcal{I}_t^{\text{fed, excl}} $$

where the wedge $\mathcal{I}\_t^{\text{fed, excl}} \setminus \mathcal{C}(s_{kt}) \neq \emptyset$ reflects imperfect public understanding. With perfect comprehension, $\mathcal{C}(s_{kt}) = \mathcal{I}_t^{\text{fed,  excl}}$, eliminating the wedge such that $\mathcal{I}_t^{\text{fed}} = \mathcal{I}_t^{text{public}}$.
Thus, a speech $s\_{kt}$ reveals a signal $\sigma$ about future states of the world $\omega\_{t+h}$ at some future horizon $h > 0$:

$$s_{kt} \supseteq \sigma(\omega\_{t+h})$$

We aim to extract some signal $\hat{\sigma}(s_{kt})$ from a speech $s_{kt}$ and forecast future macroeconomic variables over horizons $h\in\\{1,..., H\\}$. Let $\boldsymbol{x}_{t+h} \in \mathbb{R}^n$ denote the vector of macroeconomic variables at horizon $h$ which constitutes the forecast target $\omega\_{t+h} := \boldsymbol{x}^{(j)}\_{t+h}$ for some variable $j$. We index the variables to be forecast from some subset $j\in J \subseteq \\{1, ..., n\\}$. Then, the forecasting problem reads as

$$\hat{\boldsymbol{x}}^{(j)}\_{t+h} = f \left( \boldsymbol{X}^{(t)},  \hat{\sigma}(s_{kt}) \right)$$

where $f$ denotes a deep learning forecasting model described below. 
Then, the extracted signal improves forecast accuracy over forecasts not incorporating signals from Fed speeches, i.e., $\hat{\boldsymbol{x}}\_{t+h} \succ \hat{\boldsymbol{x}}_{t+h}^{text{public}}$ if and only if 

1. $\mathcal{I}_t^{\text{fed, excl}} \neq \emptyset$: the Fed possesses superior information
2. $\hat{\sigma}(s_{kt}) \neq \emptyset$: the speech reveals a signal
3. $\mathcal{I}\_t^{\text{fed, excl}} \setminus \mathcal{C}(s_{kt}) \neq \emptyset$: the public does not fully incorporate this signal.

where $\hat{\boldsymbol{x}}\_{t+h} = f \left( \boldsymbol{X}^{(t)},  \hat{\sigma}(s_{kt}) \right)$ and $\hat{\boldsymbol{x}}_{t+h}^{\text{public}} = f \left( \boldsymbol{X}^{(t)} \right)$ and $\succ$ denotes superiority under some loss function $\mathcal{L}$.


## Model Architecture

To effectively address the forecasting challenge, we propose a modeling framework centered on the Temporal Fusion Transformer (TFT)^[based on the following paper: https://arxiv.org/abs/1912.09363]. This choice is driven by the inherent complexity of our dataset, which is characterized by heterogeneous inputs including monthly macroeconomic indicators (e.g., CPI and unemployment), high-frequency exchange rates, and irregularly timed Federal Reserve speeches. Unlike traditional "black-box" deep learning models, the TFT is designed for high-performance multi-horizon forecasting while maintaining a level of interpretability that is crucial for our economic application.

**Input Processing and Signal Extraction**

The signal $\hat{\sigma}(s\_{kt})$ will be extracted from Fed speeches using state-of-the-art financial language models, such as FOMC-RoBERTa^[See https://huggingface.co/gtfintechlab/FOMC-RoBERTa], FinBERT^[ See https://huggingface.co/ProsusAI/finbert], or Central Bank RoBERTa^[See https://huggingface.co/Moritz-Pfeifer/CentralBankRoBERTa-sentiment-classifier]. These embeddings allow us to transform unstructured text into high-dimensional vectors that (potentially) capture the superior information revealed by the central bank speech, which has not already been incorporated by the public, see formal setting from above. To prevent data leakage, we will align these speech embeddings with the exact date & time they were published (tbd). 

**Heterogeneity and Interpretabilty**

A core advantage of the TFT is its internal **Variable Selection Networks**, which will allow the model to automatically weigh the importance of different inputs. it identifies whether a specific Fed speech or a static speaker characteristic (such as the specific Fed chairperson or institutional features^[See Lustenberger, Rossi and Zeitz, Central Bank Communication: New Data and Stylized Facts From a Century of Fed Speeches, forthcoming as SNB Working Paper, 2026.]) carries more predictive power for a given horizon h. Furthermore, the model incorporates **static covariates** (i.e. metadata) to provide context that remains invariant over the forecasting period, such as the specific institutional framework of the Federal Reserve. By using **Gated Residual Networks**, the TFT can selectively skip unused components of the architecture, which prevents over-parameterization.


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
- As seen in the [Temporal Fusion Transformers for Interpretable Multi-horizon Time Series Forecasting](https://arxiv.org/abs/1912.09363) paper, where the hyperparameter optimisation is conducted via random search, we will do so too. 

## Machine Learning Benchmark
- tbd.

- Model itself without text embeddings: this way, we can check whether we improve forecasts or introduce noise.

## Statistical Benchmark
- Stationarity Checks 
- AR(1) or AR(2)


## Our Questions / To Discuss

In class, we saw that we will discuss SSMs such as Mamba. Dynamic Factor SSM are particularly useful for economic datasets as it accounts for the i) short time dimension (total length and data frequency) and ii) potentially many different time series. However, we are not certain if this fits the exercise requirements. 

Alternatively, TFTs would likely be particularly useful for our research question. However, we are not certain if this fits the exercise requirements.

When using TFTs, we have the opportunity to use static covariates as inputs to our model (i.e. time invariant inputs). For example, the country of interest (=US) or, when looking at smaller samples, the Fed chairperson, e.g. -> do we have any information that is a static covariate?

<!-- 
### Questions to ask 

- Granger Causality Tests (?)
- FinBert takes fixed number of tokens as input → how do do this?
- any specific recommendations for the SSM? Mamba, e.g.?


#### Sources 

- https://arxiv.org/abs/2506.22763
- https://www.alexandria.unisg.ch/server/api/core/bitstreams/1d94cc0d-30b9-4d0d-9131-8e8c20c46837/content (finetuning FinBERT to FOMC minutes)
 -->
