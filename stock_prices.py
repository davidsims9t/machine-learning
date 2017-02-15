import pandas as pd
import quandl
import math

# Get stock price for Google
df = quandl.get('WIKI/GOOGL')

# Features
df = df[['Adj. Open','Adj. High','Adj. Low','Adj. Close','Adj. Volume']]

# Percent change
df['HL_PCT'] = (df['Adj. Low'] - df['Adj. Close']) / df['Adj. Close'] * 100.0
df['PCT_change'] = (df['Adj. Close'] - df['Adj. Open']) / df['Adj. Open'] * 100.0

df = df[['Adj. Close','HL_PCT','PCT_change','Adj. Volume']]

forecast_col = 'Adj. Close'

# If the label is undefined, set the label to NaN.
df.fillna(-9999, inplace=True)

# Predict price from 10 days ago.
forecast_out = int(math.ceil(0.01 * len(df)))

# Shift up the close price so we can show the price from 10 days from now
df['label'] = df[forecast_col].shift(-forecast_out)

print(df.head())
