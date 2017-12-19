import pandas as pd
import math
import numpy as np
from sklearn import preprocessing, cross_validation, svm
from sklearn.linear_model import LinearRegression

df = pd.read_csv('amazon.txt')

df = df[['open', 'low', 'high', 'close']]

df['hl_pct'] = (df['high'] - df['low']) / df['close'] * 100.0
df['pct_chg'] = (df['close'] - df['open']) / df['close'] * 100.0

forecast_col = 'close'
df.fillna(-9999, inplace=True)
forecast_out = int(math.ceil(0.05 * len(df)))


df['label'] = df[forecast_col].shift(-forecast_out)
df.dropna(inplace=True)

X = np.array(df.drop(['label'], 1))
y = np.array(df['label'])

X = preprocessing.scale(X)

# print(len(X))
# X = X[:-forecast_out+1]

X_train, X_test, y_train, y_test = cross_validation.train_test_split(
    X, y, test_size=0.05)

clf = LinearRegression()
# clf = svm.SVR()
clf.fit(X_train, y_train)

forecast_label = clf.predict(X[-forecast_out:])

actual_price = df['close'][-forecast_out:]
print(actual_price)
print(forecast_label)
# for items in forecast_label:
#     print(item, df[''])
