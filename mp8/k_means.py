from copy import deepcopy
import numpy as np
import pandas as pd
import sys


'''
In this problem you write your own K-Means
Clustering code.

Your code should return a 2d array containing
the centers.

'''
# Import the dataset
data = pd.read_csv("data/data/iris.data")
data = data.drop(['V5'], axis=1)
data = np.array(data)

# Make 3  clusters
k = 3

# Initial Centroids
C = [[2.,  0.,  3.,  4.], [1.,  2.,  1.,  3.], [0., 2.,  1.,  0.]]
C = np.array(C)
print("Initial Centers")
print(C)


# Choose initial clusters (C1, . . . , Ck )
# Repeat until convergence:
#     Recenter Set μj := mean(Cj) for j ∈ (1,...,k)
#     Reassign Update Cj := {xi : μ(xi) = μj} for j ∈ (1,...,k)
#     break ties arbitrarily
def k_means(C):
    delta = 1

    _C = np.array(C)
    while (delta > 1e-3):
        initial_C = _C.copy()

        _class = []
        for d in data:
            magnitudes = np.linalg.norm(d - _C, ord = 2, axis = 1)
            _class.append(np.argmin(magnitudes))

        _class = np.array(_class)

        for _c in range(len(_C)):
            cluster_data = data[np.where(_class == _c)]
            _C[_c] = np.mean(cluster_data, axis = 0)

        changes = np.linalg.norm(initial_C - _C, ord = 2, axis = 1)
        delta = np.sum(changes)

    return _C
