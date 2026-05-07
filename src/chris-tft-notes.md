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
- kafka embeddings 
- check if gdp predicts 12 months or 4 years
- (add 2nd fold) once david approves

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


### issue for david:
- tell him to checkout out holdout-validation folder, and the respective outputs
- we want to do embedinngs alternative tests
- more folds
- different alignment strategies
- different forecast horizon possibilities, which to choose? what is most scientific?
  - all predict in one go the test length (=12 years)
  - all predict with model params in steps=12 (aka 12months, or 4 years if quarterly)
- ARIMA benchmark too good, can we handicap it a bit? 