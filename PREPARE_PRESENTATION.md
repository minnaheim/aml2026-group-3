# Introduction

# Data Sources

# Embeddings

As input, we use a total of $N \approx 6000$ speeches.

## Models
For the speech embeddings, we have decided to use three models, two of which we have run so far. These are
- FinBERT: BERT-based, pre-trained on financial statements
- FOMC-RoBERTa: specifically fine-tuned on FOMC data to get hakwish/dovish classfication
- to do: LLaMA 3.1: decoder only

We have decided to pursue two embedding strategies so far.

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

# TFT

# Forecasts
Bringing it all together

## Evaluation Protocol


# Conclusion and Outlook!

