""" 
This code loads the macro variables from FRED using a FRED API key (by Minna).
Then, the code cleans the variables and, new: it also converts all variables to (mean-reverting) (growth) rates.
This is crucial for the AR/ARIMA since there, we require stationarity; 
in addition, this allows us to make the CPI and GDP forecasts more closely aligned to the UNRATE forecasts
since everything will be in terms of rates with similar magnitudes

Below, I list all vars and their transformation

CPI     log diff
PCEPI   log diff
GDP     log diff

EUR
GBP
YEN

FFR
SOFR

PAYEMS  log diff
JTSJOL  log diff
UNRATE  

T10Y2
WACLC

UMCSENT
INDPRO  log diff

"""

import pandas as pd
import numpy as np
from fredapi import Fred # see documentation here (incl. vintages) https://pypi.org/project/fredapi/
from pathlib import Path

# fred key
key = "182161f35ab1b0231ab7a21e3b991a52"
fred = Fred(key)


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

project_root = Path(__file__).resolve().parents[1]
data_dir     = project_root / "data"
data_dir.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------
# Load variables
# --------------------------------------------------------

# CPI in total
cpi = fred.get_series("CPIAUCSL")
# cpi.head()
cpi.tail()

# fed's preferred inflation metric
# Personal Consumption Expenditures Price Index
pcepi = fred.get_series('PCEPI')
pcepi.tail()

# GDP
gdp = fred.get_series('GDP')
gdp.tail()


# exchange rates daily
eur = fred.get_series('DEXUSEU')
gbp = fred.get_series('DEXUSUK')
yen = fred.get_series('DEXJPUS')


# other high-frequency vars
# (effective) federal funds rate
ffr = fred.get_series('DFF')
# Secured Overnight financing rate (SOFR) index
sofr = fred.get_series('SOFRINDEX')

# Employment stats
# Total Nonfarm Payrolls
# a measure of the number of U.S. workers in the economy
payems = fred.get_series('PAYEMS')
# Job Openings: Total Nonfarm
jtsjol = fred.get_series('JTSJOL')
# unrate
unrate = fred.get_series('UNRATE')


# some financial stuff
# 10-Year Treasury Constant Maturity Minus 2-Year Treasury Constant Maturity
t10y2y = fred.get_series('T10Y2Y')
# (Total Assets - Federal Reserve Balance Sheet)
waclc = fred.get_series('WALCL')

# sentiment
# University of Michigan Sentiment Index
umcsent = fred.get_series('UMCSENT')

# Industrial Production Index
indpro = fred.get_series('INDPRO')



# --------------------------------------------------------
# Variable transformation
# --------------------------------------------------------

cpi_logdiff = np.log(cpi).diff(1)  
pcepi_logdiff = np.log(pcepi).diff(1)
gdp_logdiff = np.log(gdp).diff(1)    
payems_logdiff = np.log(payems).diff(1)
jtsjol_logdiff = np.log(jtsjol).diff(1)
indpro_logdiff = np.log(indpro).diff(1)
waclc_logdiff = np.log(waclc).diff(1)



# --------------------------------------------------------
# Assemble and save
# --------------------------------------------------------

# assemble into quarterly, monthly, daily and weekly df
qrt_series = [gdp_logdiff]
monthly_series = [cpi_logdiff, pcepi_logdiff, payems_logdiff, jtsjol_logdiff, umcsent, indpro_logdiff, unrate]
daily_series = [eur, gbp, yen, ffr, sofr, t10y2y]
weekly_series = [waclc_logdiff]

# define names of each column
# we keep the names as before so not to break the code
qrt_names = ['GDP']
monthly_names = ['CPI', 'PCEPI', 'PAYEMS', 'JTSJOL', 'UMCSENT', 'INDPRO', 'UNRATE']
daily_names = ['EUR', 'GBP', 'YEN', 'FFR', 'SOFR', 'T10Y2Y']
weekly_names = ['WALCL']

# create df for each frequency
qrt_df = pd.concat(qrt_series, axis=1, keys=qrt_names)
monthly_df = pd.concat(monthly_series, axis=1, keys=monthly_names)
daily_df = pd.concat(daily_series, axis=1, keys=daily_names)
weekly_df = pd.concat(weekly_series, axis=1, keys=weekly_names)

# writing to csv
qrt_df.to_csv(data_dir / 'macro-vars-quarterly.csv')
monthly_df.to_csv(data_dir / 'macro-vars-monthly.csv')
daily_df.to_csv(data_dir / 'macro-vars-daily.csv')
weekly_df.to_csv(data_dir / 'macro-vars-weekly.csv')