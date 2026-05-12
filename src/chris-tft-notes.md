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

### new data
- use spot data, less include data leakage 
- include publication/vintages (how much thats possible)

# Individual To Do's 

## minna:
- check if you take the log of all vars? maybe try both?
- adapt the `data_frame_builder` with the new embeddings timeline: 1986-2023 (so holdout 6 months, then train & validation)
- shorten to only include macro data as of 2023 
- (check if gdp predicts 12 months or 4 years)
- multiple folds
- hyperparam tuning (with nested cv)


## chris
- 512 and full embeddings 
- kafka embeddings 
- adds what he did on hyperparam tuning


## anna
- trying no dim. reduction, diff. reduction, pca-with differeing, factor analysis 
- ar process adjustment 
- alignment via attention 



<!-- - max_pred 1 quarter in advance (for gdp or cpi both?) -->
- compare hyperparams
- hidden size smaller 
- more attention heads (4vs2) 



