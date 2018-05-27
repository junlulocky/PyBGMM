"""
A basic demo of 1D generated data for illustrating the PCRPMM.

Date: 2017
"""

import logging
import matplotlib.pyplot as plt
import numpy as np
import random
import sys
import collections

sys.path.append("..")

from pybgmm.prior import NIW
from pybgmm.igmm import PCRPMM
logging.basicConfig(level=logging.INFO)

random.seed(1)
np.random.seed(1)


def main():

    # Data parameters
    D = 1           # dimensions
    N = 100         # number of points to generate
    K_true = 4      # the true number of components

    # Model parameters
    alpha = 1.
    K = 3           # initial number of components
    n_iter = 40

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

    # Setup PCRPMM
    pcrpmm = PCRPMM(X, prior, alpha, save_path=None, assignments="rand", K=K)
    # pcrpmm = PCRPMM(X, prior, alpha, save_path=None, assignments="one-by-one", K=K)

    # Perform collapsed Gibbs sampling
    record_dict, distribution_dict = pcrpmm.collapsed_gibbs_sampler(n_iter, z_true, n_power=1.01, num_saved=K_true)


if __name__ == "__main__":
    main()
