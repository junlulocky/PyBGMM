import numpy as np
import sys

sys.path.append("..")

import infopy.infopy as ipy

print "## test BIC"
W = np.array([0.5, 0.5])
MU = np.array([1, 1])
SIGMA = np.array([10, 1])
N = 300 ## number of data samples
assignments = np.random.choice(MU.size, size=N, p=W)
X = np.random.normal(MU[assignments], SIGMA[assignments], size=N)
X = X[:, np.newaxis]


print "true BIC: {}".format(ipy.compute_bic(X, assignments))
assignments = np.random.permutation(assignments)
print "random BIC: {}".format(ipy.compute_bic(X, assignments))

print "## test entropy"
labels_true = np.array([1,2,3,4])
labels_pred = np.array([0,1,2,2])

## test comparison from scikit-learn
print ipy.entropy(labels_true)
print ipy.entropy(labels_pred)

## test comparison from scikit-learn
from sklearn.metrics.cluster import entropy
print entropy(labels_true)
print entropy(labels_pred)


print "## test mutual information"

print ipy.mutual_information(labels_true, labels_true)
print ipy.mutual_information(labels_pred, labels_pred)
print ipy.mutual_information(labels_true, labels_pred)

## test comparison from scikit-learn
from sklearn.metrics.cluster import mutual_info_score
print mutual_info_score(labels_true, labels_true)
print mutual_info_score(labels_pred, labels_pred)
print mutual_info_score(labels_true, labels_pred)


print "## test variation of information"
print ipy.information_variation(labels_true, labels_pred)


print "## test normalized mutual information"
print ipy.normalized_mutual_information([0, 0, 0, 0], [0, 1, 2, 3])
print ipy.normalized_mutual_information([0, 0, 1, 1], [1, 1, 0, 0])
print ipy.normalized_mutual_information([0, 0, 1, 1], [0, 0, 1, 1])

## test comparison from scikit-learn
from sklearn.metrics.cluster import normalized_mutual_info_score
print normalized_mutual_info_score([0, 0, 0, 0], [0, 1, 2, 3])
print normalized_mutual_info_score([0, 0, 1, 1], [1, 1, 0, 0])
print normalized_mutual_info_score([0, 0, 1, 1], [0, 0, 1, 1])



