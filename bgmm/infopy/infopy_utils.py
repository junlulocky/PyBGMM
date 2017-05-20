import numpy as np


def probs(x):
    """
    Computes a discrete probability distribution from a list of observations.

    :param x:
     List or numpy array
    :return:
     List: The probability of each symbol appearing in the observations from x.
    """
    n = len(x)
    num_bins = np.bincount(x)
    p = num_bins / float(n)
    return p


def cond_probs(obs):
    """
    Computes a discrete conditional probability distribution from two lists of observations.

    :param obs:
     An ordered list of observations.
    :return:
     Dict: A discrete conditional probability distribution, represented as a dictionary.
    """
    if type(obs) is str:
        obs = list(obs)
    obs = [str(ob) for ob in obs]
    syms = set(obs)
    counts = {}
    probs_dict = {}
    totals = {}
    for sym in syms:
        totals[sym] = obs.count(sym)
        counts[sym] = {}
        probs_dict[sym] = {}
        for other_sym in syms:
            counts[sym][other_sym] = 0
            probs_dict[sym][other_sym] = 0
    for i in range(len(obs) - 1):
        if obs[i + 1] in counts[obs[i]]:
            counts[obs[i]][obs[i + 1]] += 1
            continue
        counts[obs[i]][obs[i + 1]] = 1
    for sym in syms:
        div = 1 if totals[sym] is 0 else totals[sym]
        for other_sym in syms:
            probs_dict[sym][other_sym] = counts[sym][other_sym] / float(div)
    return probs_dict


def match_arrays(x, y):
    """
    Will add 0's to wichever array is shorter, until it matches the length of the longer array.
    No-op if they are the same length.

    :param x:
     Numpy array
    :param y:
     Numpy array
    :return:
     Numpy array, numpy array
    """
    if len(x) > len(y):
        for i in range(len(x) - len(y)):
            y = np.append(y, 0)
    elif len(y) > len(x):
        for i in range(len(y) - len(x)):
            x = np.append(x, 0)
    return x, y



def num_unique(x):
    """
    Returns the number of unique symbols in list/array

    :param x:
     List or numpy array.
    :return:
     Integer: the number of unique symbols in the list/array.
    """
    return len(set(x))


def bin_match(x, y):
    """
    Checks that number of unique symbols in arrays is the same.

    Examples:

    bin_match([1,2], ['A', 'B'])
    True

    bin_match([5, 6, 7], [1, 1, 4, 4])
    False

    :param x:
     List
    :param y:
     List
    :return:
     Bool
    """
    return num_unique(x) == num_unique(y)


def variable_match(x, y):
    """
    Verifies that observation variables of X and Y are from the same set.

    :param x:
     List.
    :param y:
     List.
    :return:
     Bool
    """
    return set(x) == set(y)

