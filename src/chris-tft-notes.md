### Notes by Minna on Chris' TFT Implementation
<!-- 
- why cutoff data set to 2018? 
- did you do holdout, aka train vs. validation and then one final test set? 
- rolling window 1 step ahead -> not like my prediction over entire period
- all variables are transformed to quarterly, incl. monthly and daily
- doesn't include metadata, calendar features, lag features

### for minna, to do:
- merge anna's branch back to mine
- max_pred 1 quarter in advance (for gdp or cpi both?)
- compare hyperparams
- hidden size smaller 
- more attention heads (4vs2) 

## anna's tft implementation: 
- changed variables to monthly -> ffill quarterly vars
- speaker alignment 
- with daily data, dropped basically half of the macro data, when adding the embeddings. 
-->


## future to dos:

### alternative embeddings:
- kafka embeddings (include some date. equidistant dates)
- placebo embeddings (shuffling the current embeddings)

### alignment strategies: 
- right now: means, standard deviation
- learnable parameters (but include some hardcoded things: date until FOMC or since last FOMC meeting, with euclidean/cosine distance)

### transformations:
- try with different log, diff, nothing at all, log + diff

### new data
- use spot data, less include data leakage 
- include publication/vintages (how much thats possible)
- FOMC voting rights
- FOMC dissents 
- beige book index 
- add metadata about the roles of the speaker (aligned with embeddings)

### hyperparameter tuning 
- nested cross validaiton (with darts)

### cross validation pipeline
- inlcude more than 1 fold of data!! 
- adapt the `data_frame_builder` with the new embeddings timeline: 1986-2023 (so holdout 6 months, then train & validation)

### protocolling our work
- create new markdown `presentation-preparation.md`



#### minna's todos:
- add prepare-presentation
- check if gdp predicts 12 months or 4 years
- ( kafka embeddings )
- try with different macro aggregation methods
- shorten to only include macro data as of 2023 

#### anna's todo:
- finish anna/tft
- adds FOMC 

#### chris todo:
- shuffle PCAs for embeddings
- darts implementation


<!-- - max_pred 1 quarter in advance (for gdp or cpi both?) -->
- compare hyperparams
- hidden size smaller 
- more attention heads (4vs2) 


#### notes on embeddings!

- in the data/embeddings we have both finbert and fomc-roberta embeddings
- but currently we can only use fomc-roberta, because here we have the pre-PCA version
- we need the pre-PCA version because we need to split the data into train,validation, etc. first, before we calculate the PCA (else we have data leakage)
- thus, we use `fomc-roberta/embeddings_full_*_full_fomc-roberta.csv.zip` to create a new class, which has the following methods:
    - `generate_split` (based on pre-defined split by data-frame-builder)
    - `re-calculate_PCA` on each train, test (to feed into tft)
    - `shuffle_embeddings` which shuffles within train, test (separately) to see how much chronological embeddings matter (but isnt this also data leakage?)
    - `get_embedding_data` we call this in data-frame-builder to get the data, dep. on if user passes --embedding or not.


<!-- #### notes for claude 
please build a new class into the holdout-validation which works a data-leakage free inclusion of the embeddings into the models. 

please make absolute sure, that there is no leakage present. for this work, you can use the embeddings_pipeline.py as an example, but make sure to not include the finbert and olama, just the fomc-roberta atm (this is the only embedding where we have the full embeddings, not PCAd yet.) also only do the embed_full atm, not the embed_512. 
 -->
