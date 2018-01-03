This repository contains implementations of Naive Bayes Classifier in C++ and Python.

> *Dataset used*: http://ai.stanford.edu/~amaas/data/sentiment/

----
## [C++](https://github.com/enigmaeth/machine-learning/tree/master/naive-bayes-classifier/c%2B%2B)


Contains three different implementations:
(results from a sample run are included)

- Gaussian Naive Bayes

| Case | Precision | Recall | F-Value |
| :---------: | :---------: | :---------: | :---------: |
| `Positive` | 82.828283% | 90.109890% | 86.315789% |
| `Negative` | 91.176471% | 84.545455% | 87.735849% |

- Gaussian Naive Bayes after removing stopwords

| Case | Precision | Recall | F-Value |
| :---------: | :---------: | :---------: | :---------: |
| `Positive` | 87.878788% | 90.625000% | 89.230769% |
| `Negative` | 91.089109% | 88.461538% | 89.756098% |

- Gaussian Naiva Bayes with Binary model

| Case | Precision | Recall | F-Value |
| :---------: | :---------: | :---------: | :---------: |
| `Positive` | 88.888889% | 90.721649% | 86.315789% |
| `Negative` | 91.000000% | 89.215686% | 90.099010% |


## [Python](https://github.com/enigmaeth/machine-learning/tree/master/naive-bayes-classifier/python)


There are two implementations in python.

- Gaussian Naive Bayes

This is just a simple implementation of C++ model in python, written from scratch.

- Multinomial Naive Bayes

This runs the [`sklearn.naive_bayes.MultinomialNB`](http://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.MultinomialNB.html#sklearn.naive_bayes.MultinomialNB) classifier on the same dataset.

MultinomialNB is used because the dataset was too large to be represented in raw numpy arrays. Using `scipy.sparse_matrix.csr` allowed the data to fit in memory, but scikit-learn does not support GaussianNB on sparse matrices, while MutlinomialNB's `fit()` method has params:    
`Parameters: X : {array-like, sparse matrix}, shape = [n_samples, n_features]`

### Preparation of Dataset:

The dataset contains Bag of Words for both training and testing. The following steps prepared the X vectors for MultinomialNB classifier.
 - Represent bag-of-words as a list of dicts [ `class BOWtoDictGenerator` ]
 - Picke the list of dicts for further usage
 - Using [`DictVectorizer`](http://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.DictVectorizer.html#sklearn.feature_extraction.DictVectorizer), convert each occurence dict to feature vectors with binary one-hot encoding
 - Use [`fit_tranform()`](http://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.DictVectorizer.html#sklearn.feature_extraction.DictVectorizer.fit_transform) to learn the list of feature names from DictVector
 - Convert this vector to a compressed sparse row matrix using [`sparse-csr-matrix`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csr_matrix.html#scipy-sparse-csr-matrix)
 - Pass this sparse matrix to MultinomialNB classifier's fit metohd.
 - Repeat the same for test bag-of-words, but instead of `fit-tranform`, `tranform` is required since the training dataset fixes the vocabulary. A `dimension-mismatch` error is thrown when using `fit-tranform` with test dataset.


**Results**
----
| - | True Positive | False Positive | True Negative | False Negative | Precision | Recall | F-Measure |
| :----------: | :----------: | :----------: | :----------: | :----------: | :----------: | :----------: | :----------: |
| GaussianNB | 10512 | 2487 | 10013 | 1988 | 80.87% | 84.10% | 82.45% |
| MultinomialNB | 10961 | 3121 | 10013 | 1539 | 77.84% | 87.69% | 82.47% |


