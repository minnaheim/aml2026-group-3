# Project Notes

## TO DO
### DATA
- [x] Get more variables with long time series dimension from FRED etc.
- [x] Add Composite Leading Indicator data from FRED: https://fred.stlouisfed.org/series/USALOLITONOSTSAM. The monthly series are available from January 1955 onwards. The data are compiled by the OECD (https://www.oecd.org/en/data/datasets/oecd-composite-leading-indicators-clis.html) and generally released in the first week of every month (7th-14th).
- [ ] Discuss **vintages** with David: we write about them in problemsetting but seems to be much harder (and not very ML-y) than anticipated.

### EMBEDDINGS
- [x] Test different methods => be creative
- [x] PCA should be fitted until 2011-01-01 or so only (so until the end of the very first tft training data)
- [x] Weight aggregation:
  - [x] mean and std
  - [x] exponentially decaying weights
  - [ ] attention (just one layer): once on full training, once including macro state, so somewhat 'context aware' => include description in PREPARE_PRESENTATION
  - [ ] Voter rights are included in model, what about encoding of position within Fed? (So Chair = highest rank, then Governor, then President?)
- [ ] **Normalize PCA**: probably necessary! if it were to dramatically decrease performance, ask David
- [ ] Do alternative dimensionality reductions (WIP)
  - [ ] PCA number of components => should be hyperparameter
  - [ ] No dimensionality reduction
  - [ ] Factor analysis?
- [ ] Also run using the 512 mean and CLS versions, not just full speeches   
      
### TFT
- [ ] Additional metadata from FRED
- [x] Important: also test on alternative texts once (e.g. Kafka text) to see how much TFT improvements are from speech content vs. just more data for the TFT to train on
- [ ] Hyperparameter tuning (WIP)
- [x] **Speaker characteristics!!** We can do: position in fed, year of birth, education, gender, minority, district, ...
 

### BENCHMARKS
- [x] Get AR(p) process running

### EVAL METRICS
- [x] For AR / ARIMA: expanding / rolling window
- [x] For TFT: one-holdout
  - switched to 3-fold
- [ ] **Quantile Loss/q-Risk**
- [ ] **rRMSE**, so relative RMSE error from including vs. not including speeches in TFT (can calculate from what we have)
- [ ] **variable selection**: we already have the interpret_output function but we aren't saving output yet => Save!!


### CV PIPELINE
- [x] Get up and running!!

### ABLATION
- [ ] Perform proper study
  - [ ] Include **different horizons** systematically: pipeline accomodates this, ablation code not yet

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


