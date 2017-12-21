import pandas as pd
import math
import numpy as np
from sklearn import preprocessing, cross_validation, svm
from sklearn.linear_model import LinearRegression
from tabulate import tabulate

df = pd.read_csv('breast.data')

df.replace('?', -9999, inplace=True)
df.drop(['id'], 1, inplace=True)

X = np.array(df.drop(['class'], 1))
y = np.array(df['class'])

X = preprocessing.scale(X)

X_train, X_test, y_train, y_test = cross_validation.train_test_split(
    X, y, test_size=0.4)

clf = svm.SVC()
clf.fit(X_train, y_train)

accuracy = clf.score(X_test, y_test)
print(accuracy)
