import pandas as pd
import quandl
import math
import numpy as np
from sklearn import preprocessing, cross_validation, svm
from sklearn.linear_model import LinearRegression

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

# Predict price from 32 days out.
forecast_out = int(math.ceil(0.01 * len(df)))

# Shift up the close price so we can show the price from 32 days from now
df['label'] = df[forecast_col].shift(-forecast_out)

# Drop NaN rows
df.dropna(inplace=True)

# Convert numpy array and drop label and return new data frame
X = np.array(df.drop(['label'], 1))
y = np.array(df['label'])

# Scale X along other values
X = preprocessing.scale(X)

# Make sure we only have X's where we have values for Y
y = np.array(df['label'])

# 20% of data should be testing data
X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.2)

# Fit is the same as train, score is the same as test
# Use n_jobs to enable multi-threads
clf = LinearRegression(n_jobs=-1)
clf.fit(X_train, y_train)
accuracy = clf.score(X_test, y_test)

print(accuracy)
