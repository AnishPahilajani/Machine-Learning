import pandas as pd
import quandl
import math
import numpy as np
from sklearn import preprocessing
from sklearn import model_selection
from sklearn import svm
from sklearn.linear_model import LinearRegression
import datetime
import matplotlib.pyplot as plt
from matplotlib import style
import pickle


df = quandl.get('WIKI/GOOGL')

df = df[['Adj. Open', 'Adj. High', 'Adj. Low', 'Adj. Close', 'Adj. Volume']]

df['HL_PCT'] = (df['Adj. High']-df['Adj. Close']) / df['Adj. Close'] * 100.0
df['PCT-changed'] = (df['Adj. Close']-df['Adj. Open']) / \
    df['Adj. Open'] * 100.0

# features
df = df[['Adj. Close', 'HL_PCT', 'PCT-changed', 'Adj. Volume']]

forecast_col = 'Adj. Close'
df.fillna(-9999999, inplace=True)

# forecast_out is just a percentage of our data to forecast out.
# Basically, in our dataset, maybe we want to train our model to predict the price 1% into the future.
# Then, to train, we need historically to grab values, and then use those values alongside whatever the price was
# 1% into the future (1% into the future as in 1% of the days of the entire dataset. If the dataset was 100 days,
# 1% into the future would be 1 day into the future). We use .shift, which is a pandas method, which can take a
# column and literally shift it in a direction by a number you decide. Thus, we use this to make a new column, which is
# the price column shifted, giving us
# the future prices in the same rows as current price, volume...etc to be trained against.


# price of of stock 0.01% into the future
# number of days in advace
forecast_out = int(math.ceil(0.1 * len(df)))
print(forecast_out)

# label
df['label'] = df[forecast_col].shift(-forecast_out)


# feature = X
X = np.array(df.drop(['label'], 1))
#label = y
X = preprocessing.scale(X)
#X = X[:-forecast_out]
X_lately = X[-forecast_out:]

X = X[:-forecast_out:]
df.dropna(inplace=True)
y = np.array(df['label'])

# shuffle
X_train, X_test, y_train, y_test = model_selection.train_test_split(
    X, y, test_size=0.2)

# svm.SVR(kernel='poly')  # LinearRegression(n_jobs = 10) to test LR
# clf = LinearRegression(n_jobs=10)
# clf.fit(X_train, y_train)
# with open('linearregression.pickle', 'wb') as f:
#     pickle.dump(clf, f)

pickle_in = open('linearregression.pickle', 'rb')
clf = pickle.load(pickle_in)


accuracy = clf.score(X_test, y_test)
forecast_set = clf.predict(X_lately)
print(forecast_set, accuracy, forecast_out)
df['Forecast'] = np.nan

last_date = df.iloc[-1].name
last_unix = last_date.timestamp()
one_day = 86400
next_unix = last_unix + one_day

# for loop to show dates on x axis
for i in forecast_set:
    next_date = datetime.datetime.fromtimestamp(next_unix)
    next_unix += one_day
    df.loc[next_date] = [np.nan for _ in range(len(df.columns)-1)] + [i]
df['Adj. Close'].plot()
df['Forecast'].plot()
plt.legend(loc=4)
plt.xlabel('Data')
plt.ylabel('Price')
plt.show()
