#!/usr/bin/env python

"""
A basic demo of 2D generated data for illustrating the CRPMM.

Author: Jun Lu
Contact: jun.lu.locky@gmail.com
Date: 2017
"""

import logging
import matplotlib.pyplot as plt
import numpy as np
import random
import sys
import collections

sys.path.append("..")

from bgmm.prior import NIW
from bgmm.igmm import CRPMM
from bgmm.utils.plot_utils import plot_ellipse, plot_mixture_model

logging.basicConfig(level=logging.INFO)

random.seed(1)
np.random.seed(1)


def main():

    # Data parameters
    D = 2           # dimensions
    N = 100         # number of points to generate
    K_true = 4      # the true number of components

    # Model parameters
    alpha = 1.
    K = 3           # initial number of components
    n_iter = 20

    # Generate data
    mu_scale = 4.0
    covar_scale = 0.7
    z_true = np.random.randint(0, K_true, N)
    logging.info("true clustering: {}".format(collections.Counter(z_true)))
    mu = np.random.randn(D, K_true)*mu_scale
    X = mu[:, z_true] + np.random.randn(D, N)*covar_scale
    X = X.T

    # Intialize prior
    m_0 = np.zeros(D)
    k_0 = covar_scale**2/mu_scale**2
    v_0 = D + 3
    S_0 = covar_scale**2*v_0*np.eye(D)
    prior = NIW(m_0, k_0, v_0, S_0)

    # Setup CRPMM
    crpmm = CRPMM(X, prior, alpha, save_path=None, assignments="rand", K=K)
    # crpmmmm = CRPMM(X, prior, alpha, save_path=None, assignments="one-by-one", K=K)

    # Perform collapsed Gibbs sampling
    record_dict = crpmm.collapsed_gibbs_sampler(n_iter, z_true)

    # Plot results
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plot_mixture_model(ax, crpmm)
    for k in xrange(crpmm.components.K):
        mu, sigma = crpmm.components.rand_k(k)
        plot_ellipse(ax, mu, sigma)
    plt.show()


if __name__ == "__main__":
    main()
