import pandas as pd
import os

class DataFrameBuilder:
  def __init__(self, path:  str | None = None):
    if path is None:
      self.path = os.getcwd()
    else:
       self.path = path
    self.N_FOLDS = 1 # holdout cv
    self.FINAL_HOLDOUT = 120 # 4 months
    # data without holdout
    self.TRAIN_DEC = 0.8
    self.TARGET_COLS = ["CPI", "PAYEMS", "INDPRO", "UNRATE", "GDP"]
    self.QUARTERLY_TARGETS = {"GDP"}          # native quarterly frequency
    self.MONTHLY_TARGETS   = {"CPI", "PAYEMS", "INDPRO", "UNRATE"}

  # helper to clean initial macro df, same across all frequencies
  def _read_rename_date(self, freq_path):
    df = pd.read_csv(freq_path)
    df = df.rename(columns = {'Unnamed: 0':'date'})
    df['date'] = pd.to_datetime(df['date'])
    return df


  def process_data(self):
    # ------ get daily data ---------
    daily_path = self.path + '/data/macro-vars-daily.csv'
    df_daily = self._read_rename_date(daily_path)
    df_daily = df_daily.drop(columns=["SOFR", "T10Y2Y", "EUR"]) # remove shorter vars: SOFR, T10Y2Y, EUR
    # print(df_daily.tail(20))

    # ------- get quarterly data --------

    qrtly_path = self.path + '/data/macro-vars-quarterly.csv'
    df_quartly = self._read_rename_date(qrtly_path)
    # print(df_quartly.tail(10))

    # --------- get monthly data ---------
    monthly_path = self.path + '/data/macro-vars-monthly.csv'
    df_monthly = self._read_rename_date(monthly_path)
    # remove columns which are too short 
    df_monthly = df_monthly.drop(columns=["PCEPI", "JTSJOL", "UMCSENT"])

    # merge 
    df = pd.merge(df_monthly, df_quartly, on='date', how='left')
    df = pd.merge(df_daily, df, on='date', how='left') 

    # forward-fill monthly/quarterly vars: repeat last published value until updated
    monthly_quarterly_cols = ["CPI", "PAYEMS", "INDPRO", "UNRATE", "GDP"]
    df[monthly_quarterly_cols] = df[monthly_quarterly_cols].ffill()

    # trim leading rows before daily vars (GBP, YEN) begin
    start_idx = df['GBP'].first_valid_index()
    df = df.iloc[start_idx:].reset_index(drop=True)

    print(f"\nDate range: {df['date'].min()} to {df['date'].max()}, {len(df)} rows")
    print(df.head(10))

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
# dfb = DataFrameBuilder(path)
# df = dfb.process_data()

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




