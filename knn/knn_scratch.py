import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
from collections import Counter
from math import sqrt
import warnings
style.use('fivethirtyeight')

dataset = {'k': [[1, 3], [2, 3], [3, 1]], 'r': [[6, 8], [7, 9], [8, 7]]}

new_feature = [5, 7]


# [[plt.scatter(ii[0], ii[1], s=100, color=i) for ii in dataset[i]] 
    for i in dataset]

# plt.scatter(new_feature[0], new_feature[1], s=50, color='g')
# plt.show()

def k_nearest_neighbors(data, predict, )
