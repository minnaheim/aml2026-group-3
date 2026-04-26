# Project Notes

## TO DO
### DATA
- [ ] Get more variables with long time series dimension from FRED etc.

### EMBEDDINGS
- [ ] Test different methods => be creative

### TFT
- [ ] Additional metadata from FRED

### BENCHMARKS
- [ ] Get AR(p) process running

### EVAL METRICS
- [ ] For all models: add expanding window (so far: only sliding window)


## Decisions
### Speech Data
- Merge speaker metadata to clenaed speeches: metadata only covers FRB presidents and Board members, speeches also included from deputy presidents and senior staff. We keep these speeches for now, check later if the information content maybe different.
  - Source: Central Bank Communication: New Data and Stylized Facts From a Century of Fed Speeches with Thomas Lustenberger and Enzo Rossi, under revision for the SNB Working Paper.
  - Please do not circulate until working paper available. Then, please cite.
 
### Alignment
- Keep in mind: Fed has blackout period for 10 days before FOMC
- Give calendar-weights: have the days to the next FOMC meeting as an additional input / covariate. This should work quite well since (almost) all FOMC are pre-scheduled (excep for e.g., 9/11) and thus, there shouldn't be any data leakage
  - Should be determined within TFT



 ## Things that might come up later
 - Daily data: we have business day daily data (5-day week for exchange rates, so what is traded on the market) and 7-day week (including holidays)


