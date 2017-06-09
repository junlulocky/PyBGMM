from __future__ import division

import math
import numpy as np
from infopy_utils import probs
import infopy_checker as ipchk
from numpy import array, shape, where, in1d
from scipy.spatial import distance


# Note: All functions default to log base e.

# Variables with simple values
e = 2.718281828459045
# pi = 3.141592653589793



def entropy(x, base=e):
    """calculate entropy of a list"""
    if len(x) == 0:
        return 1.0
    x = probs(x)
    total = 0
    for x_i in x:
        if x_i == 0:
            continue
        total -= x_i * math.log(x_i, base)
    return total

def normalized_mutual_information(labels_true, labels_pred, base=e):
    """
    :param labels_true:
     List or numpy array.
    :param labels_pred:
     List or numpy array.
    :param base:
     The log base used in the MI calculation. Defaults to e.
    :return:
     Float: the normalized mutual information between labels_true and labels_pred.

    Examples
    --------

    Perfect labelings are both homogeneous and complete, hence have
    score 1.0::

      >>> from sklearn.metrics.cluster import normalized_mutual_info_score
      >>> normalized_mutual_information([0, 0, 1, 1], [0, 0, 1, 1])
      1.0
      >>> normalized_mutual_information([0, 0, 1, 1], [1, 1, 0, 0])
      1.0

    If classes members are completely split across different clusters,
    the assignment is totally in-complete, hence the NMI is null::

      >>> normalized_mutual_information([0, 0, 0, 0], [0, 1, 2, 3])
      0.0
    """
    return mutual_information(labels_true, labels_pred, normalized=True, base=base)

def mutual_information(labels_true, labels_pred, normalized=False, base=e):
    """
    Compute the mutual information between two sets of observations.
    First converts observations to discrete conditional probability distribution, then computes their MI.

    :param labels_true:
     List or numpy array.
    :param labels_pred:
     List or numpy array.
    :param normalized:
     Normalize the inputs. Defaults to False.
    :param base:
     The log base used in the MI calculation. Defaults to e.
    :return:
     Float: the mutual information between labels_true and labels_pred.
    """
    labels_true, labels_pred = ipchk.check_clusterings(labels_true, labels_pred)
    labels_true = ipchk.check_numpy_array(labels_true)
    labels_pred = ipchk.check_numpy_array(labels_pred)
    numobs = len(labels_true)

    mutual_info = 0.0
    uniq_true = set(labels_true)
    uniq_pred = set(labels_pred)
    for _true in uniq_true:
        for _pred in uniq_pred:
            px = shape(where(labels_true == _true))[1] / numobs
            py = shape(where(labels_pred == _pred))[1] / numobs
            pxy = len(where(in1d(where(labels_true == _true)[0],
                                 where(labels_pred == _pred)[0]) == True)[0]) / numobs
            if pxy > 0.0:
                mutual_info += pxy * math.log((pxy / (px * py)), base)

    h_true, h_pred = entropy(labels_true), entropy(labels_pred)
    if normalized: mutual_info = mutual_info / max(np.sqrt(h_true * h_pred), 1e-10)
    # if normalized: mutual_info = mutual_info / np.log(numobs)
    return mutual_info


# Variation of information
def information_variation(labels_true, labels_pred, base=e):
    """

    :param labels_true:
     List or numpy array
    :param y:
     List or numpy array
    :param base:
     log base
    :return:
     Float: the information variation of labels_true and labels_pred
    """
    labels_true, labels_pred = ipchk.check_clusterings(labels_true, labels_pred)
    labels_true = ipchk.check_numpy_array(labels_true)
    labels_pred = ipchk.check_numpy_array(labels_pred)
    vio = entropy(labels_true, base=base) + entropy(labels_pred, base=base) - (2 * mutual_information(labels_true,
                                                                                               labels_pred,base=base))
    return vio



def compute_bic(X, assignments):

    """
    Computes the Bayesian information criterion (BIC) metric for a given clusters
    bic = 0.5 * ln(N) * k - ln(likelihood) where k is number of clusters

    :param X: multi dimentional input data, shape=(N, D), where N is sample number, D is dimension
    :param assignments: clustering assignments
    :return: BIC value
    """
    # assign centers and labels
    assignments = ipchk.check_numpy_array(assignments)
    X = ipchk.check_numpy_array(X)

    s = X.shape
    if len(s) != 2:
        raise ValueError('X must be a 2-dimensional array.')

    # assign centers and labels
    unique_assignments = np.unique(assignments)

    # unique_centers = np.zeros_like(unique_assignments)
    # for i in range(len(unique_assignments)):
    #     local_assignment = unique_assignments[i]  # get the k-th assignments
    #     x_k = X[np.where(assignments == local_assignment)[0], :]  # samples in cluster k
    #     unique_centers[i] = np.mean(x_k, axis=0)
    centers = [np.mean(X[np.where(assignments == i)[0], :], axis=0)
                      for i in range(len(unique_assignments))]

    centers = np.array(centers)



    #number of clusters
    k = len(unique_assignments)
    # size of the clusters
    n = np.bincount(assignments)

    #size of data set
    N, D = X.shape

    #compute variance for all clusters beforehand
    cl_var = [(1.0 / n[i] ) *
              sum(distance.cdist(X[np.where(assignments == i)], [centers[i]], 'euclidean')**2)
              for i in xrange(k)]

    const_term = 0.5 * k * np.log(N)

    BIC = const_term - np.sum([n[i] * np.log(n[i]) -
                         n[i] * np.log(N) -
                         ((n[i] * D) / 2) * np.log(2*np.pi) -
                         (n[i] / 2) * np.log(cl_var[i]) -
                         ((n[i] - k) / 2)
                            for i in xrange(k)])

    return BIC
