import pandas as pd
import math
import numpy as np
from sklearn import preprocessing, cross_validation, svm
from sklearn.linear_model import LinearRegression
from tabulate import tabulate

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

forecast_label_ = ['%.2f' % elem for elem in forecast_label]

actual_price = df['close'][-forecast_out:]
print(actual_price)
print(forecast_label_)
predicted_price = forecast_label.tolist()
print(predicted_price)
validate = zip(actual_price, forecast_label)
fluc = []
for i in range(0,10):
    print()

for item in validate: 
    change = round((item[1]-item[0])/item[1]*100.0, 2)
    re_ = round(item[1], 2)
    fluc.append(("%.2f" % item[0], "%.2f" % re_, str("%.2f" % change) + str('%')))

for item in fluc:
    print(item)


for i in range(0,10):
    print()