"""
Gibbs sampler for Chinese restaurant process mixture model (CRPMM)

Author: Jun Lu
Contact: jun.lu.locky@gmail.com
Date: 2017
"""

from scipy.misc import logsumexp
import numpy as np
import time
import math

from .igmm import IGMM
from ..utils import utils

class CRPMM(IGMM):

    def __init__(self, X, kernel_prior, alpha, save_path, assignments="rand", K=1, K_max=None,
            covariance_type="full"):
        super(CRPMM, self).__init__(X, kernel_prior, alpha, save_path, assignments=assignments, K=K, K_max=K_max,
            covariance_type=covariance_type)


    def collapsed_gibbs_sampler(self, n_iter, true_assignments, num_saved=3, weight_first=True):
        """
        Perform `n_iter` iterations Gibbs sampling on the CRPMM.

        A record dict is constructed over the iterations, which contains
        several fields describing the sampling process. Each field is described
        by its key and statistics are given in a list which covers the Gibbs
        sampling iterations. This dict is returned.

        Also a distribution dict is conducted when the component number equal to 'num_saved' and returned

        :param n_iter: number of sampling iterations
        :param true_assignments: true clustering assignments
        :param num_saved: save the distribution when components equal to num_saved
        :param weight_first: label switch by weight vector or by mean. By weight vector default
        :return: record dictionary & distribution dictionary
        """

        # Setup record dictionary
        record_dict = self.setup_record_dict()
        start_time = time.time()
        distribution_dict = self.setup_distribution_dict(num_saved)

        # Loop over iterations
        for i_iter in range(n_iter):

            ## Loop over data items
            # import random
            # permuted = range(self.components.N)
            # random.shuffle(permuted)
            # for i in permuted:
            for i in xrange(self.components.N):

                # Cache some old values for possible future use
                k_old = self.components.assignments[i]
                K_old = self.components.K
                stats_old = self.components.cache_component_stats(k_old)

                # Remove data vector `X[i]` from its current component
                self.components.del_item(i)

                # Compute log probability of `X[i]` belonging to each component
                log_prob_z = np.zeros(self.components.K + 1, np.float)
                # (25.35) in Murphy, p. 886
                log_prob_z[:self.components.K] = np.log(self.components.counts[:self.components.K])
                # (25.33) in Murphy, p. 886
                log_prob_z[:self.components.K] += self.components.log_post_pred(i)
                # Add one component to which nothing has been assigned
                log_prob_z[-1] = math.log(self.alpha) + self.components.cached_log_prior[i]
                prob_z = np.exp(log_prob_z - logsumexp(log_prob_z))

                # Sample the new component assignment for `X[i]`
                k = utils.draw(prob_z)
                # logger.debug("Sampled k = " + str(k) + " from " + str(prob_z) + ".")

                # Add data item X[i] into its component `k`
                if k == k_old and self.components.K == K_old:
                    # Assignment same and no components have been removed
                    self.components.restore_component_from_stats(k_old, *stats_old)
                    self.components.assignments[i] = k_old
                else:
                    # Add data item X[i] into its new component `k`
                    self.components.add_item(i, k)

            # Update record
            record_dict = self.update_record_dict(record_dict, i_iter, true_assignments, start_time)
            start_time = time.time()

        return record_dict