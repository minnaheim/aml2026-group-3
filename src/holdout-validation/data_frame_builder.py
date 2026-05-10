import pandas as pd
import numpy as np
import os
from pathlib import Path

# how many quarters to look back 
# when aggregating speeches into a single feature vector per quarter
SPEECH_WINDOW_MONTHS = 12

class DataFrameBuilder:

  # include district mapping of board and all regional districts
  # note: Board and NY always vote, others are in voting groups
  # available at: https://www.stlouisfed.org/open-vault/2022/nov/fomc-voting-rotation-explained
  DISTRICT_MAPPING = {
        "Board of Governors": "BOARD",
        "Federal Reserve Bank of New York": "NY", "Federal Reserve Bank of Atlanta": "ATL",
        "Federal Reserve Bank of Boston": "BOS", "Federal Reserve Bank of Philadelphia": "PHI",
        "Federal Reserve Bank of Richmond": "RIC", "Federal Reserve Bank of Cleveland": "CLE",
        "Federal Reserve Bank of Chicago": "CHI", "Federal Reserve Bank of St. Louis": "STL",
        "Federal Reserve Bank of Minneapolis": "MIN", "Federal Reserve Bank of Kansas City": "KC",
        "Federal Reserve Bank of Dallas": "DAL", "Federal Reserve Bank of San Francisco": "SF"
    }  

  def __init__(self, path: str | None = None):
    if path is None:
      self.path = os.getcwd()
    else:
       self.path = path
    self.N_FOLDS = 1 # holdout cv
    self.FINAL_HOLDOUT = 12 # 12 months = 1 year
    # data without holdout
    self.TRAIN_DEC = 0.8
    self.TARGET_COLS = ["CPI", "PAYEMS", "INDPRO", "UNRATE", "GDP"]
    self.QUARTERLY_TARGETS = {"GDP"} # quarterly frequency
    self.MONTHLY_TARGETS   = {"CPI", "PAYEMS", "INDPRO", "UNRATE"}

    # add fomc dissents
    self.dissent_df: pd.DataFrame | None = None

    # populated by add_leakage_free_embeddings(); None means speeches not loaded.
    self.speeches_df: pd.DataFrame | None = None
    self.pca_cols: list[str] = []


  # helper to clean initial macro df, same across all frequencies
  def _read_rename_date(self, freq_path):
    df = pd.read_csv(freq_path)
    df = df.rename(columns = {'Unnamed: 0':'date'})
    df['date'] = pd.to_datetime(df['date'])
    return df

  # add when voting
  def _is_voter(self, row):
        """Logic to determine if a speaker was a voter at time of speech."""
        year = row['Date'].year
        dist = row['District']
        
        # permanent voters
        if dist in ["BOARD", "NY"]:
            return 1
        
        # group 1: BOS, STL, KC (vote in 2028, 2025, 2022...)
        if dist in ["BOS", "STL", "KC"]:
            return 1 if (year - 2028) % 3 == 0 else 0
            
        # group 2: CLE, CHI rotate every TWO years with CHI in odd years
        if dist in ["CLE", "CHI"]:
            # Chicago votes in odd years, Cleveland in even years
            if dist == "CHI":
                return 1 if year % 2 != 0 else 0
            if dist == "CLE":
                return 1 if year % 2 == 0 else 0

        # group 3: ATL, RIC, SF (3-year cycle, 2027 ...)
        if dist in ["ATL", "RIC", "SF"]:
            return 1 if (year - 2027) % 3 == 0 else 0

        # 5. Group 4: PHI, DAL (3-year cycle, 2026 ...)
        if dist in ["PHI", "DAL", "MIN"]:
            return 1 if (year - 2026) % 3 == 0 else 0
        return 0
  
  
  def load_fomc_dissent(self, path: str | None = None) -> "DataFrameBuilder":
        """Load FOMC dissent data from Thortnon Wheelock.
        
        Columns from xlsx are: FOMC Meeting (MM.DD.YY), Dissent (Y or N),
        FOMC Votes, Votes for Action, Votes Against Action,
        No. Governors for Tighter/Easier, No. Presidents for Tighter/Easier,
        Dissenters Tighter, Dissenters Easier, Dissenters Other => all three with last names and comma separated
        
        ALSO: the file provides a complete list of fomc meeting dates which we will use 
        thus avoiding introducing an additional data source
        """
        dissent_path = path or (Path(self.path) / "data/FOMC_Dissents_Data.xlsx")
        df = pd.read_excel(dissent_path, skiprows = 3)

        # parse the MM.DD.YY date column
        df["date"] = pd.to_datetime(df["FOMC Meeting"], format="%m.%d.%y")
        df = df.sort_values("date").reset_index(drop=True)

        # normalise column names (strip whitespace)
        df.columns = df.columns.str.strip()

        # count dissents by direction
        # notes for the TAs and readers: in the FOMC, the monetary policy making body of the Fed
        # the Chair (until May 2026, Jay Powell) proposes to set rates at X, i.e., a potential Y basis points change
        # then, the governors (from the Board) and the regional presidents (with voting rights) can vote on this
        # if they dissent, they can dissent in favor of tighter policy (X_dissent > X), so a (stronger) rate hike,
        # or in favor of easier policy (X_dissent < X) THAN THE CHAIR
        # this direction (so compared to the Chair) determines the "tighter vs. easier"
        df["n_tighter"] = (
            df["No. Governors for Tighter"].fillna(0)
            + df["No. Presidents for Tighter"].fillna(0)
        ).astype(int)
        df["n_easier"] = (
            df["No. Governors for Easier"].fillna(0)
            + df["No. Presidents for Easier"].fillna(0)
        ).astype(int)
        df["n_other"]  = df["Votes Against Action"].fillna(0).astype(int) - df["n_tighter"] - df["n_easier"]
        df["n_other"]  = df["n_other"].clip(lower=0)  # guard rounding edge cases

        # net hawkish dissent: +1 per tighter dissenter, -1 per easier dissenter
        # hawkish means = favoring tighter policy
        df["dissent_net_hawk"] = df["n_tighter"] - df["n_easier"]

        # dissent intensity: share of voters who dissented
        df["dissent_rate"] = df["Votes Against Action"].fillna(0) / df["FOMC Votes"].fillna(10)

        # named dissenters
        def _count_names(cell):
            if pd.isna(cell) or str(cell).strip() == "":
                return 0
            return len([n.strip() for n in str(cell).split(",") if n.strip()])

        df["n_named_tighter"] = df["Dissenters Tighter"].apply(_count_names)
        df["n_named_easier"]  = df["Dissenters Easier"].apply(_count_names)

        self.dissent_df = df

        return self
  
  
  def _add_fomc_timing_features(self, df: pd.DataFrame) -> pd.DataFrame:
      
    """Add FOMC meeting cycle features to a monthly DataFrame.

    Features added
    --------------
    days_since_fomc   : calendar days since the last FOMC meeting as of month-start
    days_to_fomc      : calendar days until the next FOMC meeting as of month-start
    fomc_cycle_pos    : position within cycle — days_since / (days_since + days_to)
                        0 = just had a meeting, 1 = meeting is today
    meeting_this_month: 1 if an FOMC meeting falls in the same calendar month
    """
    # use meeting dates from dissents :)
    meetings = self.dissent_df["date"].sort_values().reset_index(drop=True)
        
    since, to, pos, flag = [], [], [], []

    for ts in df["date"]:
        past   = meetings[meetings <= ts]
        future = meetings[meetings >  ts]

        d_since = (ts - past.iloc[-1]).days   if len(past)   > 0 else np.nan
        d_to    = (future.iloc[0] - ts).days  if len(future) > 0 else np.nan

        cycle_pos = d_since / (d_since + d_to) if (not np.isnan(d_since) and not np.isnan(d_to) and (d_since + d_to) > 0) else np.nan

        # flag months that contain a meeting
        same_month = ((meetings.dt.year == ts.year) & (meetings.dt.month == ts.month)).any()

        since.append(d_since)
        to.append(d_to)
        pos.append(cycle_pos)
        flag.append(int(same_month))

    df["days_since_fomc"]    = since
    df["days_to_fomc"]       = to
    df["fomc_cycle_pos"]     = pos       # continuous [0, 1)
    df["meeting_this_month"] = flag      # binary

    return df
  


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
      sub = self.speeches_df.loc[mask, self.pca_cols + ['is_voter']] # keep the is voter coulmn here
      if len(sub) == 0:
          sub = self.speeches_df.loc[
              self.speeches_df["Date"] < quarter_start, self.pca_cols + ['is_voter']
          ]
      res = {}

      if len(sub) > 0:
          # new feautre: how many speeches were held be voters?
          # fixed here by moving outside loop
          res['voter_speech_ratio'] = float(sub['is_voter'].mean()) 
          for col in self.pca_cols: 
              res[f"{col}_mean"] = float(sub[col].mean()) 
              res[f"{col}_std"] = float(sub[col].std()) if len(sub) > 1 else 0.0
      else: 
          res['voter_speech_ratio'] = 0.0
          for col in self.pca_cols:
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

  def _add_dissent_features(self, df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['_month_start'] = df['date'].dt.to_period('M').dt.start_time
    month_dissent = {
        ms: self._aggregate_dissent_for_month(ms)
        for ms in df['_month_start'].unique()
    }
    for col_name in list(month_dissent.values())[0].keys():
        df[col_name] = df['_month_start'].map(lambda ms: month_dissent[ms][col_name])
    df = df.drop(columns=['_month_start'])
    return df


  def _aggregate_dissent_for_month(self, month_start: pd.Timestamp) -> dict:
    assert self.dissent_df is not None
    window_start = month_start - pd.DateOffset(months=SPEECH_WINDOW_MONTHS)
    mask = (
        (self.dissent_df["date"] >= window_start)
        & (self.dissent_df["date"] < month_start)
    )
    sub = self.dissent_df.loc[mask]

    if len(sub) == 0: # since many meetings are without dissents
        return {
            "dissent_rate_mean":     0.0,
            "dissent_net_hawk_mean": 0.0,
            "dissent_net_hawk_sum":  0.0,
            "n_tighter_sum":         0.0,
            "n_easier_sum":          0.0,
            "any_dissent_recent":    0.0,
        }
    return {
        "dissent_rate_mean":     float(sub["dissent_rate"].mean()),
        "dissent_net_hawk_mean": float(sub["dissent_net_hawk"].mean()),
        "dissent_net_hawk_sum":  float(sub["dissent_net_hawk"].sum()),
        "n_tighter_sum":         float(sub["n_tighter"].sum()),
        "n_easier_sum":          float(sub["n_easier"].sum()),
        "any_dissent_recent":    float((sub["dissent_rate"] > 0).any()),
    }
  
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
    # make 5-day daily vars (GBP, YEN) 0 on weekends instead of NaN
    df_daily[['GBP', 'YEN']] = df_daily[['GBP', 'YEN']].fillna(0)

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
    
    # add the fomc date and dissent info
    # speech features are added per-fold by add_leakage_free_embeddings()
    if self.dissent_df is not None:
        df = self._add_dissent_features(df)
        df = self._add_fomc_timing_features(df)
            
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



  def add_leakage_free_embeddings(
        self,
        splits:   list[dict],
        emb_mgr,            # EmbeddingManager instance (imported lazily to avoid circular dep)
        shuffled: bool = False,
    ) -> list[dict]:
        """Add leakage-free PCA speech features to each fold's train/test DataFrames.

        PCA is fitted on each fold's training speeches only — no look-ahead bias.
        Use this instead of loading pre-computed PCA embeddings for fomc-roberta.

        The method calls emb_mgr.generate_split() internally, then for each fold
        gets leakage-free speech embeddings, applies district mapping and voter logic,
        and runs _add_speech_features() on train and test separately.

        After this call, dfb.pca_cols contains the aggregated column names
        (pca_N_mean, pca_N_std, voter_speech_ratio) that TFTRunner expects.

        Parameters
        ----------
        splits   : as returned by generate_split()
        emb_mgr  : a loaded EmbeddingManager instance
        shuffled : if True, permute embeddings within splits (chronology ablation)
        """
        emb_mgr.generate_split(splits)
        for i, s in enumerate(splits):
            # get leakage-free speeches for this fold (0-based index into emb_mgr._splits)
            speeches_df = emb_mgr.get_embedding_data(fold=i, shuffled=shuffled)

            # district mapping and voter logic (same enrichment as load_speech_embeddings)
            if "CentralBank" in speeches_df.columns:
                speeches_df["District"] = speeches_df["CentralBank"].map(self.DISTRICT_MAPPING)
            speeches_df["is_voter"] = speeches_df.apply(self._is_voter, axis=1)

            self.speeches_df = speeches_df
            self.pca_cols    = [c for c in speeches_df.columns if c.startswith("pca_")]

            # add speech features to train and test using fold-specific PCA
            # _add_speech_features uses only speeches BEFORE each month — no look-ahead
            s["train"] = self._add_speech_features(s["train"].copy())
            s["test"]  = self._add_speech_features(s["test"].copy())
            # _add_speech_features updates self.pca_cols to aggregated names (pca_N_mean, etc.)

        return splits

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

# # with FOMC speech embeddings
# dfb = DataFrameBuilder(path).load_speech_embeddings()
# df  = dfb.process_data()   # pca_* columns now present in df

# splits, holdout = dfb.generate_split(df)

# for s in splits:
#     tr, te = s["train"], s["test"]
#     print(f"Fold {s['fold']}: train [{tr['date'].min().date()} – {tr['date'].max().date()}] ({len(tr)} rows) | "
#           f"test [{te['date'].min().date()} – {te['date'].max().date()}] ({len(te)} rows)")

# print(f"Holdout: [{holdout['date'].min().date()} – {holdout['date'].max().date()}] ({len(holdout)} rows)")


# # # quick sanity check
# tft_train = dfb.get_data(splits, fold=0, train=True,  model="TFT")

# # tft_test   = dfb.get_data(splits, fold=0, train=False, model="TFT", target= "CPI")
# # ar_train  = dfb.get_data(splits, fold=0, train=True,  model="AR",  target="CPI")
# # ar_test   = dfb.get_data(splits, fold=0, train=False, model="ARIMA",  target="GDP")

# print(f"TFT train : {len(tft_train)} daily rows   | cols: {list(tft_train.columns)}")

# print(f"TFT test : {len(tft_test)} daily rows   | cols: {list(tft_test.columns)}")
# print(f"AR  train : {len(ar_train)} monthly rows | cols: {list(ar_train.columns)}")
# print(f"ARIMA  test  : {len(ar_test)} quarterly rows  | cols: {list(ar_test.columns)}")




