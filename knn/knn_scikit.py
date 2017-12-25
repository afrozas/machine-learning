import numpy as np 
import pandas as pd
from sklearn import preprocessing, cross_validation, neighbors, svm

df = pd.read_csv('breast.data')

df.replace('?', -9999, inplace=True)
df.drop(['id'], 1, inplace=True)

X = np.array(df.drop(['class'], 1))
y = np.array(df['class'])


print(X.shape)

X = preprocessing.scale(X)
X_train, X_test, y_train, y_test = cross_validation.train_test_split(
    X, y, test_size=0.15)

clf = neighbors.KNeighborsClassifier()
clf.fit(X_train, y_train)

accuracy = clf.score(X_test, y_test)

print("KNN : ", accuracy)

example_ = np.array([[5, 1, 1, 1, 2, 1, 3, 1, 1]])
print(example_.shape)
print(clf.predict(example_))
# comparing with SVM
clf = svm.LinearSVC()
clf.fit(X_train, y_train)

accuracy = clf.score(X_test, y_test)

print("SVM Linear Classifier : ", accuracy)