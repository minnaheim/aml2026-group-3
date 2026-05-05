### Notes by Minna on Chris' TFT Implementation

- why cutoff data set to 2018? 
- did you do holdout, aka train vs. validation and then one final test set? 
- rolling window 1 step ahead -> not like my prediction over entire period

- all variables are transformed to quarterly, incl. monthly and daily
- doesn't include metadata, calendar features, lag features,



### for minna, to do:
- max_pred 1 quarter in advance (for gdp or cpi both?)
- compare hyperparams
- hidden size smaller 
- more attention heads (4vs2)



