from __future__ import division

import math
import numpy as np
from infopy_utils import probs
import infopy_checker as ipchk
from numpy import array, shape, where, in1d


# Note: All functions default to log base e.

# Variables with simple values
e = 2.718281828459045
pi = 3.141592653589793



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
def information_variation(labels_true, labels_pred):
    """

    :param labels_true:
     List or numpy array
    :param y:
     List or numpy array
    :return:
     Float: the information variation of labels_true and labels_pred
    """
    labels_true, labels_pred = ipchk.check_clusterings(labels_true, labels_pred)
    labels_true = ipchk.check_numpy_array(labels_true)
    labels_pred = ipchk.check_numpy_array(labels_pred)
    vio = entropy(labels_true) + entropy(labels_pred) - (2 * mutual_information(labels_true, labels_pred))
    return vio

