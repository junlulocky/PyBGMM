import numpy as np

def check_numpy_array(x):
    """
    Will convert x to a numpy array if it is not one already.

    :param x:
     List or numpy array.
    :return:
     Numpy array, from x.
    """
    return np.array(x)



def check_prob_sum(arr):
    """Check that the array match sum to 1"""
    return int(sum(arr)) == 1

def check_clusterings(labels_true, labels_pred):
    """Check that the two clusterings matching 1D integer arrays"""
    labels_true = np.asarray(labels_true)
    labels_pred = np.asarray(labels_pred)

    # input checks
    if labels_true.ndim != 1:
        raise ValueError(
            "labels_true must be 1D: shape is %r" % (labels_true.shape,))
    if labels_pred.ndim != 1:
        raise ValueError(
            "labels_pred must be 1D: shape is %r" % (labels_pred.shape,))
    if labels_true.shape != labels_pred.shape:
        raise ValueError(
            "labels_true and labels_pred must have same size, got %d and %d"
            % (labels_true.shape[0], labels_pred.shape[0]))
    return labels_true, labels_pred