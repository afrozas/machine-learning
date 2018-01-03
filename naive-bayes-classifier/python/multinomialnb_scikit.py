import numpy as np
import pandas as pd
import pickle
from scipy.sparse import csr_matrix
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction import DictVectorizer
from dict_generator import BOWtoDictGenerator


def get_data(file):
    """
    method to get and process test and train data
    """

    # required only once, at next runs pickle can be used
    bow_dict = BOWtoDictGenerator().load(file)
    with open("bow_dict.pickle", "wb") as b:
        pickle.dump(bow_dict, b)

    # uncomment when running second time
    # loads the static bag or words from pickle
    with open("bow_dict.pickle", "rb") as b:
        bow_dict = pickle.load(b)

    return bow_dict


def train(clf, vec):
    bow_dict = get_data('../dataset/train/labeledBow.feat')
    csr_vec = vec.fit_transform(bow_dict)
    X = csr_matrix(csr_vec)
    y = [0]*12500 + [1]*12500 
    clf.fit(X, y)
    print("Training Completed")


def test(clf, vec):
    test_vectors_dict = get_data('../dataset/test/labeledBow.feat')
    csr_vec = vec.transform(test_vectors_dict)
    test_vectors = csr_matrix(csr_vec)
    pred_ratings = clf.predict(test_vectors)
    true_ratings = [0]*12500 + [1]*12500

    test_index = 25000
    tp, fp, tn, fn = 0, 0, 0, 0
    for i in range(0, test_index):
        if true_ratings[i] == 1 and pred_ratings[i] == 1:
            tp += 1
        elif true_ratings[i] == 0 and pred_ratings[i] == 1:
            fp += 1
        if true_ratings[i] == 0 and pred_ratings[i] == 0:
            tn += 1
        if true_ratings[i] == 1 and pred_ratings[i] == 0:
            fn += 1

    precision = float(tp)/(tp+fp)
    recall = float(tp)/(tp+fn)
    print("tp fp tn fn : ", tp, fp, tn, fn)
    print("Precision : %.2f%%" % (precision*100))
    print("Recall : %.2f%%" % (recall*100))
    print("F Measure : %.2f%%" % (float(2*precision*recall)/(precision+recall)*100))


def main():
    clf = MultinomialNB()
    vec = DictVectorizer()    
    train(clf, vec)
    test(clf, vec)


if __name__ == '__main__':
    main()
