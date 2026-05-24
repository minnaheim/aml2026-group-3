# Introduction

# Data Sources and the Data Frame Builder

For our analysis, we use three different data types. First, we compile a macroeconomic dataset containing variables such as CPI, GDP and the unemployment rate from the US. All data is taken from FRED. Second, we use a set of Federal Reserve Speeches: both the Bank for International Settlements (BIS) and a recent paper have collected a set of speeches since 1986. Third, we complement these speeches with a variety of speaker and FOMC specific data. This includes speaker characteristics such as their position within the Fed, their highest obtained education, age and gender, but also aspects such as the exogenous FOMC rotation of voting rights (in place since the 1940s) and dissents within the FOMC.

The macroeconomic data is preprocessed: variables such as GDP and CPI which are variables with a clear time trend are log-differenced. As a consequence, they are stationary which is crucial for our statistical benchmarks. Further, log-differencing brings all transformed variables to a similar scale (approximate growth rates), which improves numerical stability and training dynamics of the TFT. More details are listed below.

As we will discuss below, we will employ different strategies to embed the speeches. First, I would like to focus on the alignment strategy.

*Note: our working paper will be coming out in the next few weeks. I (Anna) will be spending the summer figuring out how to clean the speeches to a level that we could use them for our analysis. Then, we would have data since 1914 which will allow us to increase the number of folds.*

## Alignment

Since most macroeconomic variables, in particular those which are the ones we are interested in forecasting, are of monthly frequency, we decide to also pursue this approach; we have tested using a daily alignment strategy, however, too many covariates are then not changing over 30 or 90 (for GDP) days which significantly hampered performance.

First, some points on the alignment of the macroeconomic variables: for GDP which is the only lower frequency variable (quarterly), we forward-fill all months without a known value. For all variables which are of higher frequency, we take the mean and the standard deviation over the month. This reflects both the overall level and the volatility, the latter of which often reflects times of economic turmoil if it increases.

The alignment of the speeches is a more crucial point. We have a total of three methods which we describe in detail below. First, we use a rolling window  mean aggregation. For each month $m$, we take the speeches of the last X months (defined by variable SPEECH_WINDOW_MONTHS which we treat as a hyperparameter) and compute the mean and standard deviation of all speech dimensions (more below). We complement this with information on how many of these speeches were held by voters, given their influence in the FOMC.

Second, we use exponential decaying weights such that speeches closer to month $m$ receive a higher weight. **We also treat the degree of decay as a hyperparameter.**

Third, we use attention-based aggregation- Each speech has key (PCA embeddings) and query (macro state in month of forecast) such that we ask: given where we are now, which past speeches are most informative? For example, speeches discussing price stability in the past will thus receive higher weights for the forecasting. Then, we calculate the attention weigths.

# Embeddings

As input, we use a total of $N \approx 6000$ speeches.

## Models
For the speech embeddings, we have decided to use two models, namely
- FinBERT: BERT-based, pre-trained on financial statements
- FOMC-RoBERTa: specifically fine-tuned on FOMC data to get hakwish/dovish classfication

### Truncation
Assumption: The highest informational value is right at the beginning of a speech. 

For this, we truncate each speech to the first 512 tokens. For FinBERT and FOMC-RoBERTa, we extract both a CLS token embeddings, that is, the hidden state of the CLS token which has attended to all other tokens and aggregated their information, as well as a mean-pooled embedding which averages the hidden states of all tokens.

**However, we have decided to drop this: the reason is that the title itself and the introductory words by the speaker already encompass more than 512 tokens such that this strategy does not recover substantial information. Having also spoken to several speech writers (e.g., Kevin Kliesen from the St. Louis Fed), the main information is normally given over the entire speech.**

### Chunk and Averaging
Assumption: Speeches start off with many introductory words and are structured to provide an overall picture. This is driven by the fact that the audience is typical well educated on economic matters, e.g. at academic conferences.

To account for the token limit of the transformers, we split each speech into 512-token windows with an overlap of 128-tokens, i.e., a stride of 384 tokens. These 512 tokens include a CLS and SEP token at the beginning and end of the chunk, respectively for the encoder model, allowing the model to treat each chunk as an independent document. This is crucial since both FinBERT and FOMC-RoBERTa were trained this way. The chunk size while running is 16. Then, we embed each windows separately and average the vectors per speech. 

## Dimensionality Reduction

Both FinBERT and FOMC-RoBERTa embeddings are of dimension $768$ for each speech. Thus, we employ two different methods for dimensionality reduction and, in addition, try to not perform any dimensionality reduction. **Note however that no dimensionality reduction cannot be computed?**

First, we use principal component analysis (PCA) to reduce the informational content to X components, treated as a hyperparameter. For example, 20 components cover approximately 80% of the overall variance from the embeddings.
We fit the PCA on the entire set of speeches *in the training data*, using random state 42 for reproducibility. Through this cutoff, we can exclude concerns of data leakage.

Second, we use factor analysis (FA), again using X components which we treat as a hyperparameter.

*Note: the PCA basically uses the covariance matrix from which one calculates the Eigenvalues and chooses the Eigenvectors with the largest Eigenvalue, so a Eigendecomposition. These provide the PCA coordinates.*

*Note: Factor Analysis models the observed embeddings as a linear combination of a smaller number of latent factors plus noise. Unlike PCA which maximises explained variance, FA assumes a generative model. The parameters are estimated via maximum likelihood, iterating until convergence (EM algorithm). The key difference from PCA: FA explicitly models the noise term of the model, meaning it focuses on the shared variance between dimensions rather than total variance.*

# Benchmarks

## AR(1) Process

A standard benchmark in macroeconomic forecasting is an AR(p) process, i.e., a univariate autoregressive model. We closely align the prediction method with the TFT approach to make the forecasting results comparable. The order of autoregressivity $p$ is selected for each variable individually with an upper limit of $p=3$ using the Akaike Information Criterion (AIC). The fallback is an AR(1) process.

For the prediction, we follow the TFT which only has access to its own predictions (plus the training data) and never the actual test observations for the multi-step forecasting. Thus, the AR(p) process performs a forecast for a window of $h$ monthly steps. Then, it appends these predictions to the context and then forecasts the next window using the expanded context. Thus, the coefficients are fixed and only the context is updated, making it a fair comparison to the TFT.


## ARIMA

To expand on the simple AR(p) benchmark, we also fit an ARIMA to the macroeconomic variables. Since we have already ensured the stationarity of the macroeconomic variables, we can keep the integration order $d=0$. Then, we use auto arima to find the best lag order, subject to maximal lags of the autoregressive and moving average component, i.e., $p, q \in \{0,1,2,3\}$. The model chooses the optimal ARIMA(p,d=0,q) process based on the AIC. We also allow the model to identify seasonality for the monthly and quarterly variables. As a fallback if the data is insufficient, so fewer than 24 monthly or 8 quarterly observations, we set an ARIMA(1,0,1). For computational reasons, we cache results; as long as the training data remains the same (meaning the same cutoff date), the results of auto arima are reused.

The prediction method is identical to the AR(p) process.

# TFT

The Temporal Fusion Transformer (TFT) is a model developed to perform well for multi-horizon forecasting, by including not only the data which is to be forecast (which is what ARIMA, AR do), but it allows the user to include a variety of different data types, which can be used to improve forecasts. When looking at the `pytorch-forecasting` implementation of the Temporal Fusion Transformer, we can see the following parameters:

```
static_categoricals – names of static categorical variables

static_reals – names of static continuous variables

time_varying_categoricals_encoder – names of categorical variables for encoder

time_varying_categoricals_decoder – names of categorical variables for decoder

time_varying_reals_encoder – names of continuous variables for encoder

time_varying_reals_decoder – names of continuous variables for decoder

categorical_groups – dictionary where values are list of categorical variables that are forming together a new categorical variable which is the key in the dictionary
```

In our case, we use these parameters in the following way: 

```
static_categoricals=['series_id'] + meta_cat_cols,
static_reals=meta_real_cols,
time_varying_known_reals=['time_idx', 'day_of_week', 'week_of_year', 'month', 'is_holiday',
                                      # fomc dates known in advance — only included with speeches
                                      *fomc_known_present],
time_varying_unknown_reals=[*covariates, *lag_cols, *pca_cols_present, *dissent_cols_present],
```
Meaning, we use: 

- as categorical variables which are static (=time-invariant) our metadata from our macro variables such as the frequency of the data (M, Q, or D) and the units of the variabe, and the id of the series. 

- For our static real variables we use the real-valued metadata, aka how long the series exists, how popular it is (how many downloads). 

- for our real variables that are time variant, we use some metainformation about the days which the macro vars are published, aka what week of the year it was, whether the day was a holiday, etc. we also use the information about the time to the next FOMC meeting, how many FOMC meetings there were this month, etc. 

- for our time variying unknown reals (which are real values which we dont know in advance, unlike week of the year) we use the macro variables, their lags, the PCA columns of the embeddings (if included) and how much dissent there was within FOMC members during that meeting.

## Preprocessing

We compute monthly lags at horizons of 1, 2, 6, and 12 months for all five target variables (CPI, PAYEMS, INDPRO, UNRATE, GDP). For quarterly GDP the lag unit is one quarter (3 months) rather than one month. The lags are computed via a merge-asof on calendar dates rather than row offsets, which is important because the data is at monthly frequency with forward-filled rows: a row-based shift would give the wrong date. Rows in which the 12-month lag is still undefined (the first ~12 months of the series) are dropped.

The target is normalized using an `EncoderNormalizer` with a softplus transformation. This normalizer is computed from the encoder context window of each sample, so it is inherently local and avoids any global scaling that would introduce look-ahead bias.

## Training Protocol

The training split is itself divided into a fitting window and a validation window: the last `MAX_PREDICTION_LENGTH = 12` rows of the training data are held out as a true out-of-sample validation set. This means the early stopping criterion (monitored on `val_loss`) reflects genuine out-of-sample fit within the training period rather than in-sample fit. Early stopping has patience 5 and a minimum improvement threshold of 0.01; the best checkpoint is saved and reloaded for inference. Training runs for at most 50 epochs. Optionally, training curves can be logged to Weights & Biases via a `WandbLogger`.

## Inference: Rolling-Window Prediction

The test period is typically longer than `MAX_PREDICTION_LENGTH`, so predictions are generated in non-overlapping 12-month windows. For each window, the context passed to the model is the full training set concatenated with all test months up to the current window. Crucially, any test observations that fall within the context are replaced by the model's own previous predictions, never by the true realised values — the same assumption made by the AR and ARIMA benchmarks. This ensures the multi-step forecasts are genuinely out-of-sample and that the three models are evaluated on the same information set.

After inference, results are resampled to native frequency (monthly for most targets, quarterly for GDP) and saved alongside the benchmark predictions.

# Forecasts
Bringing it all together

# Hyperparameter Tuning:

After discussing the topic with David, he told us to fix all of the hyperparams we can think of today, and use bayesian optimisation. 

We want to train our **two TFT models**:

-> TFT without speech embeddings (serves as our ML-benchmark)

-> TFT with speech embeddings

to predict these two categories:
 
1. Targets ["CPI", "GDP", "UNRATE"]
2. Horizons [3, 6, 12]

Meaning, when it comes to hyperparameter tuning, we need to tune the following hyperparameters for these 12 model combinations, i.e.:

1. TFT Macro on CPI with horizon=3 
2. TFT with Speech Embeddings on CPI with horizon=3 
3. TFT Macro on CPI with horizon=6
etc. 

## Defining our Hyperparams we want to tune: 

We defined out hyperparameter tuning strategy to work in two steps:

1. First, we tune all of the non-context relevant paramters, then fix those
2. Second, train the econ-specific context-parameters (embedding params)


### First Part of HP Tuning:

- `max_encoder_length` — context window, e.g. [12-48] months
- `lstm_layers` — {1, 2, 4} (currently 4; was 2)
- `hidden_size` — {16, 32, 64, 128, 256} 
- `hidden_continuous_size` — {2, 4, 8, 16}
- `dropout` — between [0.05, 0.55] 
- `learning_rate` — between [1e-4, 0.1]
- `batch_size` — {16, 32, 64, 128} (currently 128 val, 16 train)
- `max_epochs` — fixed at 50 (tuned via early stopping, not a sweep param)
- **normalizer** (per target, since series differ in scale/stationarity):
    - `EncoderNormalizer(transformation="None")`  — no transform (after getting new macro data)  
    - `GroupNormalizer` — global scaling across the group; tested before, currently dropped

### Second Part of HP Tuning 

- `SPEECH_WINDOW_MONTHS` — rolling look-back for speech aggregation, e.g. [3-12] 
- `N_PCA` between [5,30]
- **embedding type**:
    - FinBERT (chunk-mean)
    - FOMC-RoBERTa (chunk-mean)
- **dimensionality reduction of speeches**:
     - (Speech Embeddings as is (no dim reduction))
     — PCA components per speech `N_PCA`, e.g. [5-30] 
     - Factor Analysis also `N_PCA`(just use same factor)
- **embedding aggregator**:
    - mean 
    - (exponential) decay
    - attention-context-based


## Evaluation Protocol

To avoid any look-ahead bias in model selection, we follow a strict two-stage evaluation protocol. In the first stage, we run a three-fold expanding-window cross-validation on the pre-holdout data (1986–2022) to tune hyperparameters and compare models. Each fold adds roughly one block of observations to the training window while the subsequent non-overlapping block serves as the test window. All hyperparameter tuning (architecture search in Stage 1, embedding parameter search in Stage 2) is performed exclusively on these CV folds.

In the second and final stage, we retrain each model — AR, ARIMA, and TFT — on the **entire** pre-holdout period and evaluate on the **true holdout set**: the final 12 months of data (~January 2023 – December 2023). This holdout window is held completely separate throughout and is never used for training, order selection, or hyperparameter tuning. The final evaluation uses the same metrics as the CV stage (MAE, RMSE, and quantile loss for TFT's prediction intervals), and reports one result per model per target. This mirrors standard practice in macroeconomic forecasting where the final evaluation is a clean out-of-sample exercise.


# Conclusion and Outlook!

