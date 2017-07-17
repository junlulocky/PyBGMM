
import random
import numpy as np
from scipy.spatial import distance
from scipy.cluster.vq import vq

def draw(p_k):
    """
    Draw from a discrete random variable with mass in vector `p_k`.

    Indices returned are between 0 and len(p_k) - 1.
    :param p_k: probability vector
    :return random choice from probability vector
    """
    k_uni = random.random()
    for i in xrange(len(p_k)):
        k_uni = k_uni - p_k[i]
        if k_uni < 0:
            return i
    return len(p_k) - 1

def draw_rand(p_k):
    """
    TODO: change me
    :param p_k:
    :return:
    """
    p_k = p_k/np.sum(p_k)
    return np.random.choice(np.arange(0,len(p_k)), p=p_k)

def cluster_loss_inertia(x, assignments):
    """
    compute squared root loss for all data (all clusters)
    :param x: all data
    :param assignments: all assignments
    :return: loss
    """
    unique_assignments = np.unique(assignments)
    unique_centers = np.zeros_like(unique_assignments) # 'deprecated': store center for each cluster
    unique_dist = np.zeros_like(unique_assignments)  # store the loss for each cluster

    for i in range(len(unique_assignments)):
        local_assignment = unique_assignments[i]  # get the k-th assignments
        x_k = x[np.where(assignments==local_assignment)[0],:]  # samples in cluster k
        # unique_centers[i], unique_dist[i] = compute_mean_dist(x_k)
        unique_dist[i] = compute_dist(x_k)

    loss = np.sum(unique_dist)
    return loss



def compute_dist(x):
    """
    computer squared root loss for 'x', all 'x' belong to one cluster
    :param x: data
    :return: loss for 'x'
    """
    if x.shape[1] == 1:
        ## one dimensional case
        mean = 0
        cnt = 0.0
        for point in x:
           mean += point
           cnt += 1.0
        mean = mean/cnt

        dist = np.sqrt(np.sum(np.square(x - mean)))
    elif x.shape[1] == 2:
        ## two dimensional case
        mean = np.array([0.,0.])
        cnt = 0.0
        for i in range(x.shape[0]):  # loop over data
            mean[0] += x[i,0]
            mean[1] += x[i,1]
            cnt += 1.0
        mean[0] = mean[0] / cnt
        mean[1] = mean[1] / cnt

        # dist = np.sqrt(np.sum(np.square(x - mean)))
        dist = np.sqrt(np.sum(np.square(x - mean)))
    else:
        mean = np.mean(x, axis=0)
        dist = np.sqrt(np.sum(np.square(x - mean)))


    # return mean, dist
    return dist


def compute_mean(x):
    """
    computer squared root loss for 'x', all 'x' belong to one cluster
    :param x: data
    :return: loss for 'x'
    """
    if x.shape[1] == 1:
        ## one dimensional case
        mean = 0
        cnt = 0.0
        for point in x:
           mean += point
           cnt += 1.0
        mean = mean/cnt

        dist = np.sqrt(np.sum(np.square(x - mean)))
    elif x.shape[1] == 2:
        ## two dimensional case
        mean = np.array([0.,0.])
        cnt = 0.0
        for i in range(x.shape[0]):  # loop over data
            mean[0] += x[i,0]
            mean[1] += x[i,1]
            cnt += 1.0
        mean[0] = mean[0] / cnt
        mean[1] = mean[1] / cnt

        # dist = np.sqrt(np.sum(np.square(x - mean)))
        dist = np.sqrt(np.sum(np.square(x - mean)))

    return mean
    # return dist

def compute_bic(x, assignments):
    """
    Computes the BIC metric for a given clusters

    Parameters:
    -----------------------------------------
    kmeans:  List of clustering object from scikit learn

    X     :  multidimension np array of data points

    Returns:
    -----------------------------------------
    BIC value
    """
    # assign centers and labels
    unique_assignments = np.unique(assignments)
    unique_centers = np.zeros_like(unique_assignments)
    for i in range(len(unique_assignments)):
        local_assignment = unique_assignments[i]  # get the k-th assignments
        x_k = x[np.where(assignments==local_assignment)[0],:]  # samples in cluster k
        unique_centers[i] = compute_mean(x_k)

    centers = unique_centers
    labels = assignments

    # centers = [kmeans.cluster_centers_]
    # labels  = kmeans.labels_

    #number of clusters
    m = len(unique_assignments)
    # size of the clusters
    n = np.bincount(labels)
    #size of data set
    N, d = x.shape

    #compute variance for all clusters beforehand
    cl_var = (1.0 / (N - m) / d) * \
             sum(
        [sum(distance.cdist(x[np.where(labels == i)], [np.array([centers[i]])], 'euclidean')**2)
         for i in range(m)]
             )

    const_term = 0.5 * m * np.log(N) * (d+1)

    BIC = np.sum([n[i] * np.log(n[i]) -
               n[i] * np.log(N) -
             ((n[i] * d) / 2) * np.log(2*np.pi*cl_var) -
             ((n[i] - 1) * d/ 2) for i in range(m)]) - const_term

    return(BIC)