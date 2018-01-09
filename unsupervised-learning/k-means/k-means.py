import numpy as np
import pandas as pd
import pickle
from scipy.sparse import csr_matrix
from sklearn.cluster import KMeans
from sklearn.feature_extraction import DictVectorizer
from dict_generator import BOWtoDictGenerator

global cluster


def get_data(file):
    """
    method to get and process test and train data
    """

    # required only once, at next runs pickle can be used
    # bow_dict = BOWtoDictGenerator().load(file)
    # with open("bow_dict.pickle", "wb") as b:
    #     pickle.dump(bow_dict, b)

    # uncomment when running second time
    # loads the static bag or words from pickle
    with open("bow_dict.pickle", "rb") as b:
        bow_dict = pickle.load(b)

    return bow_dict


def train(cluster, vec):
    # bow_dict = get_data('../naive-bayes-classifier/dataset/train/labeledBow.feat')
    # csr_vec = vec.fit_transform(bow_dict)
    # X = csr_matrix(csr_vec)
    # cluster.fit(X)
    # print("Training Completed")

    # with open("kmeans.pickle", "wb") as b:
    #     pickle.dump(cluster, b)

    with open("kmeans.pickle", "rb") as b:
        cluster = pickle.load(b)
        print("Cluster loaded from pickle")

    centroids = cluster.cluster_centers_
    labels = cluster.labels_    
    print("centroids : ", centroids)
    print("*******")
    print("labels : ", labels)
    return cluster


def test(cluster, vec):
    pred_ratings = cluster.labels_
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
        
    switch_temp = fp
    fp = tn
    tn = switch_temp
    precision = float(tp)/(tp+fp)
    recall = float(tp)/(tp+fn)
    print("tp fp tn fn : ", tp, fp, tn, fn)
    print("Precision : %.2f%%" % (precision*100))
    print("Recall : %.2f%%" % (recall*100))
    print("F Measure : %.2f%%" % (float(2*precision*recall)/(precision+recall)*100))


def main():
    cluster = KMeans(n_clusters=2)
    vec = DictVectorizer()    
    cluster = train(cluster, vec)
    test(cluster, vec)


if __name__ == '__main__':
    main()
