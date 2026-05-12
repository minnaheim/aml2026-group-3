# Introduction

# Data Sources and the Data Frame Builder

For our analysis, we use three different data types. First, we compile a macroeconomic dataset containing variables such as CPI, GDP and the unemployment rate from the US. All data is taken from FRED. Second, we use a set of Federal Reserve Speeches: both the Bank for International Settlements (BIS) and a recent paper have collected a set of speeches since 1986. Third, we complement these speeches with a variety of speaker and FOMC specific data. This includes speaker characteristics such as their position within the Fed, their highest obtained education, age and gender, but also aspects such as the exogenous FOMC rotation of voting rights (in place since the 1940s) and dissents within the FOMC.

As we will discuss below, we will employ different strategies to embed the speeches. First, I would like to focus on the alignment strategy.

*Note: our working paper will be coming out in the next few weeks. I (Anna) will be spending the summer figuring out how to clean the speeches to a level that we could use them for our analysis. Then, we would have data since 1914 which will allow us to increase the number of folds.*

## Alignment

Since most macroeconomic variables, in particular those which are the ones we are interested in forecasting, are of monthly frequency, we decide to also pursue this approach; we have tested using a daily alignment strategy, however, too many covariates are then not changing over 30 or 90 (for GDP) days which significantly hampered performance.

First, some points on the alignment of the macroeconomic variables: for GDP which is the only lower frequency variable (quarterly), we forward-fill all months without a known value. **We are aware that there are some spot forecasts of GDP in the US. One approach we are yet to implement is to actually include a gap to account for this.** For all variables which are of higher frequency, we simply take the last value of the month. **Anna: My suggestion would be to actually take the last value AND extract the standard deviation over the month. This could signal increased volatility, in particular for the yield spreads.** 

The alignment of the speeches is a more crucial point. Currently, we use somewhat of a rolling window aggregation. For each month $m$, we take the speeches of the last 12 months and compute the mean and standard deviation of each of the 20 PCA dimensions (more below). We complement this with information on how many of these speeches were held by voters, given their influence in the FOMC. **Anna: I think the 12 month look-back is not really helpful for forecasting the business cycle component. We should definitely treat the SPEECH_WINDOW_MONTHS variable as a hyperparameter during the cross-validation! This significantly strengthens the argument for more than 1 cross-validation fold.  My guess is that a shorter horizon will improve forecasting!**

### Alignment Of Speeches: Possible Ideas

**ANNA: most importantly, we need to switch the "calculating monthly mean and std of PCA components". This is definitely misplaced.**
As the most straightforward way, I suggest to use exponential decaying weights, so: the closer the speech is to month $m$, the higher the weight. 

A more innovative idea would be the following: speeches given close around FOMC meetings are more informative. Also, the voting status and the position within the Fed (Chair, Governor, President) definitely matters. We could somehow combine these three aspects, maybe multiplicatively. So, mathematically, we would have something like, each PCA of speech $s$ receives weight $w_s$ such that

$$w_s = w_s^{vote} \times w_s^{fomc} \times w_s^{position}$$

where $w_s^{vote} = 1$ for non-voters and $>1$ for voters; $w_s^{fomc} = \exp(-\lambda d_s^{since}) + exp(-\lambda d_s^{to})$, reflecting that days since and to the next FOMC matter and speeches right around an FOMC meeting are likely more informative. Finally, I think it's best if we don't overengineer the position weight. Instead, simply set Chair = 3, Board = 2 and President = 1.

Even more innovative but possibly a bit beyond this project: use attention-based aggregation, so: speech has key (PCA embeddings) and query (macro state in month of speech). then, calculate attention weights. Intuitively, that would mean that if inflation is high, speeches which talk about price stability and not unemployment get higher weights. I think this might be hard given our data limitations. If we were to continue the project with the speeches from my paper, so since 1914, approx. 10k, maybe it would work.

# Embeddings

As input, we use a total of $N \approx 6000$ speeches.

## Models
For the speech embeddings, we have decided to use three models, two of which we have run so far. These are
- FinBERT: BERT-based, pre-trained on financial statements
- FOMC-RoBERTa: specifically fine-tuned on FOMC data to get hakwish/dovish classfication
- to do: LLaMA 3.1: decoder only

We have decided to pursue two embedding strategies so far.

Please note, that although these 3 embeddings exist, only the FOMC-RoBERTa is used for the `holdout-validation` pipeline atm. The reason for this, is because currently only the FOMC-RoBERTa embeddings exist in full (without being PCA'd) and the others have been PCA'd already. So even if we split the embeddings to only include the training data, there is data leakage anyhow (similar to David's example in the lecture, with the normalising of data pre-split). 


### Truncation
Assumption: The highest informational value is right at the beginning of a speech. 

For this, we truncate each speech to the first 512 tokens. For FinBERT and FOMC-RoBERTa, we extract both a CLS token embeddings, that is, the hidden state of the CLS token which has attended to all other tokens and aggregated their information, as well as a mean-pooled embedding which averages the hidden states of all tokens.

### Chunk and Averaging
Assumption: Speeches start off with many introductory words and are structured to provide an overall picture. This is driven by the fact that the audience is typical well educated on economic matters, e.g. at academic conferences.

To account for the token limit of the transformers, we split each speech into 512-token windows with an overlap of 128-tokens, i.e., a stride of 384 tokens. These 512 tokens include a CLS and SEP token at the beginning and end of the chunk, respectively for the encoder model, allowing the model to treat each chunk as an independent document. This is crucial since both FinBERT and FOMC-RoBERTa were trained this way. The chunk size while running is 16. Then, we embed each windows separately and average the vectors per speech. 

## Dimensionality Reduction

Both FinBERT and FOMC-RoBERTa embeddings are of dimension $768$ for each speech. We use principal component analysis (PCA) to reduce the informational content to 20 components which covers about 80% of the overall variance from the embeddings. This helps in two ways, namely first, with 20 PCA, the speeches do not dominate the macro variables and second, since the embedding dimensions are highly correlated, including all 768 dimensions does not increase informational value but only adds noise and is inefficient. The idea is that the principal components then reflect a hawkish/dovish tone, the degree of uncertainty, etc. until 20 components are reached.

We fit the PCA on the entire set of speeches, using random state 42 for reproducibility. We acknowledge that fitting the PCA on the entire data set introduces data leakage since the principal components on the later training data are also calculated on speeches that are only available during test time. 

**Comment Anna: I see two ways to deal with this. Either, we acknowledge it and say, well whatever can we do. Alternatively, we only fit the PCA on data until our very first cutoff point with the TFT (since it's not relevant for AR and ARIMA anyway) and for all speeches afterwards, we apply the PCA from the training data. I definitely prefer the latter version which luckily does not introduce additional compute time.** 

As we will discuss during the TFT, we aggreate the embeddings up; this alignment of the speeches which occur in irregular intervals is a major point during the TFT training.

*Note: the PCA basically uses the covariance matrix from which one calculates the Eigenvalues and chooses the Eigenvectors with the largest Eigenvalue, so a Eigendecomposition. These provide the PCA coordinates.*

*Note 2: do we want to increase PCA a bit, to maybe 30?*

Thus, our final output and therefore, input into the TFT, are 20 principal components for each speech, i.e., $N\times 20$.

# Benchmarks

## AR(1) Process

A standard benchmark in macroeconomic forecasting is an AR(1) process, i.e., a univariate autoregressive model. We closely align the prediction method with the TFT approach to make the forecasting results comparable.

During training, we first transform the data such that each series is stationary. That means for CPI and GDP, we use a log-diff, yielding the effective growth rate, and we directly take levels for the unemployment rate. An augmented Dickey-Fuller test confirms that the series are stationary. 

For the prediction, we follow the TFT which only has access to its own predictions (plus the training data) and never the actual test observations for the multi-step forecasting. Thus, the AR(1) process performs a forecast for a window of 12 steps. Then, it appends these predictions to the context and performs the subsequent 12 steps of the prediction using the training + the first forecasted window. Thus, the coefficients are fixed and only the context is updated, making it a fair comparison to the TFT.

Since we will be showing the results not in the transformed space but in the original space, we invert the results for the log-differenced data by cumsum-exponatiating to the original units in levels.

## ARIMA

To expand on the simple AR(1) benchmark, we also fit an ARIMA to the macroeconomic variables. As before, we are handling stationarity by taking log-differences of GDP and CPI which allows us to keep the integration order $d=0$. Then, we use auto arima to find the best lag order, subject to maximal lags of the autoregressive and moving average component, i.e., $p, q \in \{0,1,2,3\}$. The model chooses the optimal ARIMA(p,d=0,q) process based on the Akaike Information Criterion. We also allow the model to identify seasonality for the monthly and quarterly variables. As a fallback if the data is insufficient, so fewer than 24 monthly or 8 quarterly observations, we set an ARIMA(1,0,1). For computational reasons, we cache results; as long as the training data remains the same (meaning the same cutoff date), the results of auto arima are reused.

The prediction method is identical to the AR(1) process.

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

Before building the dataset, we apply two preprocessing steps. First, skewed macroeconomic variables (CPI, PAYEMS, INDPRO, GDP) are log-transformed so that the lags computed from them are also in log space. This keeps the scale consistent with the target and avoids the TFT having to learn the transformation implicitly. Second, we compute monthly lags at horizons of 1, 2, 6, and 12 months for all five target variables (CPI, PAYEMS, INDPRO, UNRATE, GDP). For quarterly GDP the lag unit is one quarter (3 months) rather than one month. The lags are computed via a merge-asof on calendar dates rather than row offsets, which is important because the data is at monthly frequency with forward-filled rows: a row-based shift would give the wrong date. Rows in which the 12-month lag is still undefined (the first ~12 months of the series) are dropped.

The target is normalized using an `EncoderNormalizer` with a softplus transformation. This normalizer is computed from the encoder context window of each sample, so it is inherently local and avoids any global scaling that would introduce look-ahead bias.

## Architecture and Hyperparameters

**IMPORTANT** : this is all subject to change, havent done hyperparam tuning yet! 


We use the `pytorch-forecasting` implementation of the TFT with the following hyperparameters:

| Parameter | Value | Rationale |
|---|---|---|
| `max_encoder_length` | 24 months | 2-year context window |
| `max_prediction_length` | 12 months | 1-year forecast horizon, matching AR/ARIMA |
| `lstm_layers` | 4 | deeper sequence encoding |
| `hidden_size` | 64 | main model width |
| `attention_head_size` | 2 | multi-head self-attention in decoder |
| `dropout` | 0.2 | regularisation |
| `hidden_continuous_size` | 8 | embedding width for continuous variables |
| `output_size` | 1 | point forecast |
| `loss` | SMAPE | scale-independent, comparable across targets |
| `learning_rate` | 0.03 | |
| `gradient_clip_val` | 0.25 | prevents exploding gradients |

## Training Protocol

The training split is itself divided into a fitting window and a validation window: the last `MAX_PREDICTION_LENGTH = 12` rows of the training data are held out as a true out-of-sample validation set. This means the early stopping criterion (monitored on `val_loss`) reflects genuine out-of-sample fit within the training period rather than in-sample fit. Early stopping has patience 5 and a minimum improvement threshold of 0.01; the best checkpoint is saved and reloaded for inference. Training runs for at most 50 epochs. Optionally, training curves can be logged to Weights & Biases via a `WandbLogger`.

## Inference: Rolling-Window Prediction

The test period is typically longer than `MAX_PREDICTION_LENGTH`, so predictions are generated in non-overlapping 12-month windows. For each window, the context passed to the model is the full training set concatenated with all test months up to the current window. Crucially, any test observations that fall within the context are replaced by the model's own previous predictions, never by the true realised values — the same assumption made by the AR and ARIMA benchmarks. This ensures the multi-step forecasts are genuinely out-of-sample and that the three models are evaluated on the same information set.

After inference, log-transformed targets (CPI, PAYEMS, INDPRO, GDP) are back-transformed via exponentiation. Results are then resampled to native frequency (monthly for most targets, quarterly for GDP) and saved alongside the benchmark predictions.

# Forecasts
Bringing it all together

## Evaluation Protocol


# Conclusion and Outlook!

