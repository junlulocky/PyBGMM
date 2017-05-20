"""
Gibbs sampler for powered Chinese restaurant process mixture model (PCRPMM)

Author: Jun Lu
Contact: jun.lu.locky@gmail.com
Date: 2017
"""

from scipy.misc import logsumexp
import numpy as np
import time
import math
import logging
import matplotlib.pyplot as plt

from .igmm import IGMM
from ..utils import utils
from ..utils.plot_utils import plot_ellipse, plot_mixture_model

logger = logging.getLogger(__name__)

class PCRPMM(IGMM):

    def __init__(self, X, prior, alpha, save_path, assignments="rand", K=1, K_max=None,
            covariance_type="full"):
        super(PCRPMM, self).__init__(X, prior, alpha, save_path, assignments=assignments, K=K, K_max=K_max,
            covariance_type=covariance_type)


    # @profile
    def collapsed_gibbs_sampler(self, n_iter, true_assignments,
                                 n_power=1.01, power_burnin=0,
                                 num_saved=3, weight_first=True, flag_power=True):
        """
        Perform `n_iter` iterations Gibbs sampling on the PCRPMM.

        A record dict is constructed over the iterations, which contains
        several fields describing the sampling process. Each field is described
        by its key and statistics are given in a list which covers the Gibbs
        sampling iterations. This dict is returned.

        Also a distribution dict is conducted when the component number equal to 'num_saved' and returned
        :param n_iter: number of sampling iterations
        :param true_assignments: true clustering assignments
        :param flag_power: flag for pCRP, if False, it is equal to CRPMM
        :param n_power: power value
        :param power_burnin: iteration<power_buruin will set power value to 1 (i.e. CRPMM)
                             iteration>power_burnin wil set power value =  n_power (i.e. PCRPMM)
        :param num_saved: save the distribution when components equal to num_saved
        :param weight_first: label switch by weight vector or by mean. By weight vector default
        :return: record dictionary & distribution dictionary
        """


        # Setup record dictionary
        record_dict = self.setup_record_dict()
        start_time = time.time()
        distribution_dict = self.setup_distribution_dict(num_saved)


        ## TODO used for noise analysis
        all_noise_data = []

        # Loop over iterations
        for i_iter in range(n_iter):
            # print 'iter: {}'.format(i_iter)

            # isNoiseAnalysis = False
            # if isNoiseAnalysis:
            #     # logging.info('clusters:{}'.format(self.components.counts[:self.components.K]))
            #     small_cluster_idx = np.where(self.components.counts[:self.components.K] <= 1)[0]
            #     # logging.info('less than 2:{}'.format(small_cluster_idx))
            #     # logging.info('assignments: {}'.format(collections.Counter(self.components.assignments)))
            #
            #     data_idx = [i for i, row in enumerate(self.components.assignments) if row in small_cluster_idx]
            #     logging.info("data idx:{}".format(data_idx))
            #
            #     all_noise_data = all_noise_data + data_idx
            #     logging.info("unique idx:{}".format(np.unique(all_noise_data)))

            ## save the wanted distribution
            if num_saved == self.components.K and i_iter > 1:
                distribution_dict = self.update_distribution_dict(distribution_dict, weight_first)




            if flag_power and n_power > 1:
                if i_iter % 20 == 0:
                    logger.info(" Permutate data; " + "Power value: {}".format(n_power))
                data_loop_list = np.random.permutation(xrange(self.components.N))
            else:
                data_loop_list = xrange(self.components.N)
            ## Loop over data items
            for i in data_loop_list:

                # Cache some old values for possible future use
                k_old = self.components.assignments[i]
                K_old = self.components.K
                stats_old = self.components.cache_component_stats(k_old)

                # Remove data vector `X[i]` from its current component
                self.components.del_item(i)

                # Compute log probability of `X[i]` belonging to each component
                log_prob_z = np.zeros(self.components.K + 1, np.float)
                if flag_power and i_iter > power_burnin:
                    ## for pCRP
                    log_prob_z[:self.components.K] = np.log(
                        np.power(self.components.counts[:self.components.K], n_power))
                else:
                    ## plain gibbs sampling
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
            ## end loop data

            ## TODO noise proof
            isNoiseProof = False
            if isNoiseProof:
                noise_useful_cluster_num = []
                noise_nonuseful_cluster_num = []
                for i_cluster in range(self.components.K):
                    if self.components.counts[i_cluster] == 1:
                        noise_nonuseful_cluster_num.append(i_cluster)
                    else:
                        noise_useful_cluster_num.append(i_cluster)

                small_cluster_idx = np.where(self.components.counts[:self.components.K] == 1)[0]

                small_data_idx = [i for i, row in enumerate(self.components.assignments) if
                                  row in small_cluster_idx]

                ## Loop over data items
                for i in small_data_idx:

                    # Cache some old values for possible future use
                    k_old = self.components.assignments[i]
                    K_old = self.components.K
                    stats_old = self.components.cache_component_stats(k_old)

                    # Remove data vector `X[i]` from its current component
                    self.components.del_item(i)

                    # Compute log probability of `X[i]` belonging to each component
                    log_prob_z = np.zeros(self.components.K + 1, np.float)

                    ## plain gibbs sampling
                    # (25.35) in Murphy, p. 886
                    log_prob_z[:self.components.K] = np.log(self.components.counts[:self.components.K])
                    # (25.33) in Murphy, p. 886
                    log_prob_z[:self.components.K] += self.components.log_post_pred(i)
                    # Add one component to which nothing has been assigned
                    log_prob_z[-1] = math.log(self.alpha) + self.components.cached_log_prior[i]
                    prob_z = np.exp(log_prob_z - logsumexp(log_prob_z))

                    k = utils.draw(prob_z)
                    while k in small_cluster_idx:
                        k = utils.draw(prob_z)

                    # Add data item X[i] into its component `k`
                    if k == k_old and self.components.K == K_old:
                        # Assignment same and no components have been removed
                        self.components.restore_component_from_stats(k_old, *stats_old)
                        self.components.assignments[i] = k_old
                    else:
                        # Add data item X[i] into its new component `k`
                        self.components.add_item(i, k)
            ## end noise proof


            # Update record
            record_dict = self.update_record_dict(record_dict, i_iter, true_assignments, start_time)
            start_time = time.time()

        return record_dict, distribution_dict