# Project Notes

## TO DO
### DATA
- [ ] Get more variables with long time series dimension from FRED etc.

### EMBEDDINGS
- [ ] Test different methods => be creative
- [ ] PCA should be fitted until 2011-01-01 or so only (so until the end of the very first tft training data)
  - so for pre-2011 speeches, use fit_transform
  - for everything later, use transform only
  - RERUN: Implement in data_frame_builder and rerun with cutoff on 2011-01-01 => i mistakenly ran it with cutoff 2014-01-01
- [ ] Normalize PCA: probably necessary! if it were to dramatically decrease performance, ask David
      
### TFT
- [x] Additional metadata from FRED
- [ ] Important: also test on alternative texts once (e.g. Kafka text) to see how much TFT improvements are from speech content vs. just more data for the TFT to train on

### BENCHMARKS
- [x] Get AR(p) process running

### EVAL METRICS
- [x] For AR / ARIMA: expanding / rolling window
- [x] For TFT: one-holdout
- [x] unified metrics for all


### CV PIPELINE
- [x] Get up and running!! (WIP)

# Notes from Meeting on Tue, 12.5 (individual todos)

## minna:
- [x] multiple folds
- [ ] hyperparam tuning (with nested cv) -> chris
- [ ] why are the predictions always linear? look at AR weight matrices


## chris
- [ ] 512 and full embeddings 
- [ ] kafka embeddings 
- [ ] adds what he did on hyperparam tuning


## anna
- [ ] trying no dim. reduction, diff. reduction, pca-with differeing, factor analysis 
- [ ] ar process adjustment 
- [ ] alignment via attention 
- [ ] use the growth rates instead of level


## Decisions
### Data Sources
- Merge speaker metadata to clenaed speeches: metadata only covers FRB presidents and Board members, speeches also included from deputy presidents and senior staff. We keep these speeches for now, check later if the information content maybe different.
  - Source: Central Bank Communication: New Data and Stylized Facts From a Century of Fed Speeches with Thomas Lustenberger and Enzo Rossi, under revision for the SNB Working Paper.
  - Please do not circulate until working paper available. Then, please cite.
 
- Beige Book Index: economic sentiment index derived via text analysis of Federal Reserve Beige Books by Gascon, Charles S and Martorana, Joseph, Quantifying the Beige Book’s ‘Soft’ Data, 2025
- FOMC Dissents Data: record of dissents on FOMC monetary policy votes from 1936 to 2025 (continuously updated) by Thornton, Daniel L and Wheelock, David C, Making Sense of Dissents: A History of FOMC Dissents, 2014.
- Tealbook (formerly Greenbook): projections for many of the variables also forecast in the Federal Reserve Bank of Philadelphia's Survey of Professional Forecasters
  - However: only published five years after the FOMC Meeting, so not relevant for our forecasting purpose after all
  - https://www.philadelphiafed.org/surveys-and-data/real-time-data-research/greenbook
 
### Alignment
- Keep in mind: Fed has blackout period for 10 days before FOMC
- Give calendar-weights: have the days to the next FOMC meeting as an additional input / covariate. This should work quite well since (almost) all FOMC are pre-scheduled (excep for e.g., 9/11) and thus, there shouldn't be any data leakage
  - Should be determined within TFT
 



 ## Things that might come up later
 - Daily data: we have business day daily data (5-day week for exchange rates, so what is traded on the market) and 7-day week (including holidays)
 - Potentially have a gap between train and test data to account for the fact that some data series are published with a lag but there are spot forecasts etc.
  - At least, this is the case in Switzerland and many other European countries, in particular for GDP


