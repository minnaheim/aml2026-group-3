import pandas as pd
import numpy as np
import os
from pathlib import Path

# how many quarters to look back 
# when aggregating speeches into a single feature vector per quarter
SPEECH_WINDOW_MONTHS = 12

class DataFrameBuilder:
  def __init__(self, path:  str | None = None):
    if path is None:
      self.path = os.getcwd()
    else:
       self.path = path
    self.N_FOLDS = 1 # holdout cv
    self.FINAL_HOLDOUT = 12 # 12 months = 1 year
    # data without holdout
    self.TRAIN_DEC = 0.8
    self.TARGET_COLS = ["CPI", "PAYEMS", "INDPRO", "UNRATE", "GDP"]
    self.QUARTERLY_TARGETS = {"GDP"}          # native quarterly frequency
    self.MONTHLY_TARGETS   = {"CPI", "PAYEMS", "INDPRO", "UNRATE"}
    
    # populated by load_speech_embeddings(); None means speeches not loaded.
    self.speeches_df: pd.DataFrame | None = None
    self.pca_cols: list[str] = []


  # helper to clean initial macro df, same across all frequencies
  def _read_rename_date(self, freq_path):
    df = pd.read_csv(freq_path)
    df = df.rename(columns = {'Unnamed: 0':'date'})
    df['date'] = pd.to_datetime(df['date'])
    return df
  
  # load speech embeddings based on chris's logic
  def load_speech_embeddings(
        self,
        embedding_subpath: str = "data/embeddings/fomc-roberta/"
                                 "embeddings_pca_mean_full_fomc-roberta.csv",
    ) -> "DataFrameBuilder":
        """Load FOMC speech PCA embeddings and cache them on the instance.
 
        Must be called before process_data() if goal to merge speech features
        into the main DataFrame. Returns self so it can be chained:
 
            dfb = DataFrameBuilder(path).load_speech_embeddings()
            df  = dfb.process_data()
 
        The loaded data is stored as:
            self.speeches_df  — full embedding DataFrame sorted by Date
            self.pca_cols     — sorted list of "pca_N" column names
 
        Parameters
        ----------
        embedding_subpath :
            Path to the PCA CSV relative to self.path.
        """
        speech_path = Path(self.path) / embedding_subpath
        if not speech_path.exists():
            raise FileNotFoundError(
                f"Speech embedding file not found: {speech_path}\n"
                "Check the path or omit load_speech_embeddings() for macro-only mode."
            )
        df = pd.read_csv(speech_path, parse_dates=["Date"])
        df = df.sort_values("Date").reset_index(drop=True)
        self.speeches_df = df
        self.pca_cols    = sorted(c for c in df.columns if c.startswith("pca_"))
        print(
            f"[speeches] Loaded {len(df)} speeches "
            f"with {len(self.pca_cols)} PCA dims "
            f"({df['Date'].min().date()} → {df['Date'].max().date()})"
        )
        return self
      
      
  def _aggregate_speeches_for_month(self, quarter_start: pd.Timestamp) -> dict[str, float]:
      """Mean PCA vector for speeches in the rolling window before *month_start*.

      Window: [quarter_start - SPEECH_WINDOW_QUARTERS quarters, quarter_start).
      Falls back to any speech before quarter_start if the window is empty.
      """
      assert self.speeches_df is not None
      window_start = quarter_start - pd.DateOffset(months=SPEECH_WINDOW_MONTHS)
      mask = (
          (self.speeches_df["Date"] >= window_start)
          & (self.speeches_df["Date"] < quarter_start)
      )
      sub = self.speeches_df.loc[mask, self.pca_cols]
      if len(sub) == 0:
          sub = self.speeches_df.loc[
              self.speeches_df["Date"] < quarter_start, self.pca_cols
          ]
      res = {}
      for col in self.pca_cols:
          if len(sub) > 0:
              res[f"{col}_mean"] = float(sub[col].mean())
              res[f"{col}_std"] = float(sub[col].std()) if len(sub) > 1 else 0.0
          else:
              res[f"{col}_mean"] = 0.0
              res[f"{col}_std"] = 0.0
      return res
 
  def _add_speech_features(self, df: pd.DataFrame) -> pd.DataFrame:
      """Broadcast month-level speech PCA aggregates onto every daily row.

      Called at the end of process_data() when speeches_df is available.
      One aggregation call per unique quarter keeps this fast on daily data.
      No look-ahead: each quarter gets the PCA vector computed from speeches
      that occurred *before* that quarter's start date.
      """
      df = df.copy()
      df['_month_start'] = df['date'].dt.to_period('M').dt.start_time
      month_pca = {
            ms: self._aggregate_speeches_for_month(ms)
            for ms in df['_month_start'].unique()
        }
      # Extract new keys (pca_0_mean, pca_0_std, etc.)
      sample_keys = list(month_pca.values())[0].keys()
      for col_name in sample_keys:
          df[col_name] = df['_month_start'].map(lambda ms: month_pca[ms][col_name])
          
      # Update self.pca_cols to include the new suffixes for the TFTRunner
      self.pca_cols = list(sample_keys)
      
      df = df.drop(columns=['_month_start'])
      return df

  def process_data(self):
    # SWITCH TO MONTHLY FREQUENCY AS BASE!!
    # --------- get monthly data ---------
    monthly_path = self.path + '/data/macro-vars-monthly.csv'
    df_monthly = self._read_rename_date(monthly_path)
    # remove columns which are too short 
    df_monthly = df_monthly.drop(columns=["PCEPI", "JTSJOL", "UMCSENT"])
    
    # ------ get daily data ---------
    daily_path = self.path + '/data/macro-vars-daily.csv'
    df_daily = self._read_rename_date(daily_path)
    df_daily = df_daily.drop(columns=["SOFR", "T10Y2Y", "EUR"]) # remove shorter vars: SOFR, T10Y2Y, EUR
    # print(df_daily.tail(20))

    # ------- get quarterly data --------

    qrtly_path = self.path + '/data/macro-vars-quarterly.csv'
    df_quartly = self._read_rename_date(qrtly_path)
    # print(df_quartly.tail(10))
    
    # START with monthly as the base
    df = df_monthly.copy()
    
    # merge 
    df = pd.merge(df, df_quartly, on='date', how='left')
    
    # resample Daily variables to Monthly before merging
    # to prevent the 1,200 row expansion
    df_daily_m = df_daily.set_index("date").resample("MS").last().reset_index()
        
    df = pd.merge(df, df_daily_m, on='date', how='left') 

    # forward-fill quarterly vars (GDP)
    df[self.TARGET_COLS] = df[self.TARGET_COLS].ffill()
    
    # ------ speech embeddings (optional) -------
    # only if load_speech_embeddings() was called beforehand.
    if self.speeches_df is not None:
        df = self._add_speech_features(df)
        
    # trim leading NaNs
    df = df.dropna(subset=['GBP']).reset_index(drop=True)

    print(f"\nMonthly Date range: {df['date'].min()} to {df['date'].max()}, {len(df)} rows")    
    # print(df.head(10))
    return df
  

   
  def generate_split(self, df):
    df_cv = df.iloc[:-self.FINAL_HOLDOUT].reset_index(drop=True)
     # first remove final holdout, not part of cv!
    df_holdout = df.iloc[-self.FINAL_HOLDOUT:].reset_index(drop=True)

    splits = []
    # rolling window cv
    for i in range(self.N_FOLDS):
        train_end = (i + 1) * int(len(df_cv)*self.TRAIN_DEC)
        test_end = train_end + int(len(df_cv)*(1-self.TRAIN_DEC))
        splits.append({
            "fold": i + 1,
            "train": df_cv.iloc[:train_end],
            "test": df_cv.iloc[train_end:test_end],
        })

    return splits, df_holdout



  def get_data(self, splits, train: bool= True,  model: str = "TFT", target: str = "CPI", fold = 0):
    """
    fold   : 1 fold only atm
    train  : True → training split, False → test split
    model  : "AR" or "ARIMA" → univariate series resampled to native frequency
             "TFT"           → full daily dataframe
    target : which macro variable to forecast (only used for AR / ARIMA)
    """
    assert target in self.TARGET_COLS, f"target must be one of {self.TARGET_COLS}"
    df_split = splits[fold]["train" if train else "test"]

    if model in ("AR", "ARIMA"):
        freq = "QS" if target in self.QUARTERLY_TARGETS else "MS"
        return (df_split.set_index("date")[[target]]
                        .resample(freq).first()
                        .dropna()
                        .reset_index())

    return df_split.copy()  # TFT: full daily df


  
# this is how you would try it out

# path = "/Users/minna/Code/FS26/AML/aml2026-group-3"

# # marco-only, so as before
# dfb = DataFrameBuilder(path)
# df  = dfb.process_data()
#
# # with FOMC speech embeddings
# dfb = DataFrameBuilder(path).load_speech_embeddings()
# df  = dfb.process_data()   # pca_* columns now present in df

# splits, holdout = dfb.generate_split(df)

# for s in splits:
#     tr, te = s["train"], s["test"]
#     print(f"Fold {s['fold']}: train [{tr['date'].min().date()} – {tr['date'].max().date()}] ({len(tr)} rows) | "
#           f"test [{te['date'].min().date()} – {te['date'].max().date()}] ({len(te)} rows)")

# print(f"Holdout: [{holdout['date'].min().date()} – {holdout['date'].max().date()}] ({len(holdout)} rows)")


# # quick sanity check
# tft_train = dfb.get_data(splits, fold=0, train=True,  model="TFT")
# tft_test   = dfb.get_data(splits, fold=0, train=False, model="TFT", target= "CPI")
# ar_train  = dfb.get_data(splits, fold=0, train=True,  model="AR",  target="CPI")
# ar_test   = dfb.get_data(splits, fold=0, train=False, model="ARIMA",  target="GDP")

# print(f"TFT train : {len(tft_train)} daily rows   | cols: {list(tft_train.columns)}")
# print(f"TFT test : {len(tft_test)} daily rows   | cols: {list(tft_test.columns)}")
# print(f"AR  train : {len(ar_train)} monthly rows | cols: {list(ar_train.columns)}")
# print(f"ARIMA  test  : {len(ar_test)} quarterly rows  | cols: {list(ar_test.columns)}")




