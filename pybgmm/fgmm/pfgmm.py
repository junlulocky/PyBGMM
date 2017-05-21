"""
Base class for Finite Gaussian mixture model (GGMM)

Author: Jun Lu
Contact: jun.lu.locky@gmail.com
Date: 2017
"""

from scipy.misc import logsumexp
from scipy.special import gammaln
import logging
import numpy as np
import time
from scipy import stats
import copy
import math

from ..gaussian.gaussian_components import GaussianComponents
from ..gaussian.gaussian_components_diag import GaussianComponentsDiag
from ..gaussian.gaussian_components_fixedvar import GaussianComponentsFixedVar

from ..utils import utils
from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn.metrics import mutual_info_score

from ..ars.ars import ARS
from fgmm_hyperprior_func import *
from ..infopy.infopy import information_variation

from .fgmm import FGMM

logger = logging.getLogger(__name__)


#-----------------------------------------------------------------------------#
#                                 FGMM CLASS                                 #
#-----------------------------------------------------------------------------#

class PFGMM(FGMM):
    """
    Bayesian inference for a POWERED finite Gaussian mixture model (PFGMM).

    See `GaussianComponents` or `GaussianComponentsDiag` for an overview of the
    parameters not mentioned below.

    Parameters
    ----------
    alpha : float
        Concentration parameter for the symmetric Dirichlet prior over the
        mixture weights.
    K : int
        The number of mixture components. This is actually a maximum number,
        and it is possible to empty out some of these components.
    assignments : vector of int or str
        If vector of int, this gives the initial component assignments. The
        vector should therefore have N entries between 0 and `K`. Values of
        -1 is also allowed, indicating that the data vector does not belong to
        any component. Alternatively, `assignments` can take one of the
        following values:
        - "rand": Vectors are assigned randomly to one of `K` components.
        - "each-in-own": Each vector is assigned to a component of its own.
    covariance_type : str
        String describing the type of covariance parameters to use. Must be
        one of "full", "diag" or "fixed".
    """

    def __init__(
            self, X, kernel_prior, dir_prior, K, assignments="rand",
            covariance_type="full"
            ):
        super(PFGMM, self).__init__(X, kernel_prior, dir_prior, K, assignments=assignments,
            covariance_type=covariance_type)


    def powered_chinese_restaurant(self, n_iter, _true_assignment, n_power=1.1,
                                   weight_first=True):
        return self.constrained_reassign_gibbs_sample(n_iter, _true_assignment,
                                                      flag_constrain=False, n_constrain=100000, thres=0,
                                                      flag_power=True, n_power=n_power,
                                                      flag_loss=False, n_loss_step=100000, flag_marg=False,
                                                      loss_burnin=1000000, )

    def constrained_sampling(self, n_iter, _true_assignment, n_constrain=20, thres=0.04,
                             weight_first=True):
        return self.constrained_reassign_gibbs_sample(n_iter, _true_assignment,
                                                      flag_constrain=True, n_constrain=n_constrain, thres=thres,
                                                      flag_power=False, n_power=1,
                                                      flag_loss=False, n_loss_step=100000, flag_marg=False,
                                                      loss_burnin=1000000,
                                                      weight_first=weight_first)

    def plain_gibbs_sampling(self, n_iter, _true_assignment,
                             weight_first=True):

        return self.constrained_reassign_gibbs_sample(n_iter, _true_assignment,
                                          flag_constrain=False, n_constrain=1, thres=0.,
                                          flag_power=False, n_power=1,
                                          flag_loss=False, n_loss_step=100000, flag_marg=False,
                                                      loss_burnin=1000000,
                                                      weight_first=weight_first)


    def loss_based_sampling(self, n_iter, _true_assignment,
                             n_loss_step=20, flag_marg=True, loss_burnin=500,
                            weight_first=True):

        return self.constrained_reassign_gibbs_sample(n_iter, _true_assignment,
                                          flag_constrain=False, n_constrain=1, thres=0.,
                                          flag_power=False, n_power=1,
                                          flag_loss=True, n_loss_step=n_loss_step, flag_marg=flag_marg,
                                                      loss_burnin=loss_burnin,
                                                      weight_first=weight_first)

    def ada_power(self, n_iter, _true_assignment, r_up=1.3, adapower_perct=0.04,
                                             adapower_burnin=500,
                                             weight_first=True):
        return self.constrained_reassign_gibbs_sample(n_iter, _true_assignment,
                                                    flag_adapower=True, r_up=r_up,
                                                      adapower_perct=adapower_perct,
                                                    adapower_burnin=adapower_burnin,
                                                      weight_first=weight_first)

    def constrained_reassign_gibbs_sample(self, n_iter, true_assignments,
                                          flag_constrain=False, n_constrain=1, thres=0.04,
                                          flag_power=False, n_power=1,
                                          flag_loss=False, n_loss_step=20, flag_marg=True, loss_burnin=500,
                                          flag_adapower=False, r_up=1.3, adapower_perct=0.04,
                                             adapower_burnin=500,
                                          weight_first=True):
        """
        Perform `n_iter` iterations Gibbs sampling on the FBGMM.

        A record dict is constructed over the iterations, which contains
        several fields describing the sampling process. Each field is described
        by its key and statistics are given in a list which covers the Gibbs
        sampling iterations. This dict is returned.
        """

        # Setup record dictionary
        record_dict = self.setup_record_dict()
        start_time = time.time()
        distribution_dict = self.setup_distribution_dict(self.K, n_iter)


        constrain_thres = self.components.N * thres


        # if self.hyperprior is not None:
        #     distribution_dict["alpha"] = np.zeros(shape=(n_iter,))


        ## Loop over iterations
        for i_iter in range(n_iter):

            if flag_constrain:
                if i_iter % n_constrain == 0:
                    logging.info('performing constrain step')
                    logging.info('all cluster nk: {}'.format(self.components.counts[:self.components.K]))
                    isConstrained = True
                    tmp_useful_cluster_num = []
                    tmp_nonuseful_cluster_num = []
                    for i_cluster in range(self.components.K):
                        if self.components.counts[i_cluster] > constrain_thres:
                            tmp_useful_cluster_num.append(i_cluster)
                        else:
                            tmp_nonuseful_cluster_num.append(i_cluster)
                else:
                    isConstrained = False

            if flag_loss and i_iter % n_loss_step == 0 and i_iter > loss_burnin:
                logging.info('cluster nk before loss sampling: {}'.format(self.components.counts[:self.components.K]))
                copy_components = copy.deepcopy(self.components)
                if flag_marg:
                    max_prob = float('-inf')
                    max_prob_components = copy_components
                else:
                    min_loss = float('+inf')
                    min_loss_components = copy_components

                loss_cnt = 0
                while copy_components.K > 2:
                    loss_cnt += 1
                    if loss_cnt > 50:
                        break
                    # print "iter: {}".format(i_iter)
                    # print "1: {}".format(copy_components.K)

                    # because we need to assign the copy to max_components
                    copy_components = copy.deepcopy(copy_components)

                    loss_nonuseful_cluster_idx = np.argmin(copy_components.counts[:copy_components.K])
                    loss_useful_cluster_num = []

                    for i_cluster in range(copy_components.K):
                        if i_cluster != loss_nonuseful_cluster_idx:
                            loss_useful_cluster_num.append(i_cluster)
                    # tmp_counts = copy_components.counts[:copy_components.K]
                    # tmp_counts[loss_nonuseful_cluster_idx] = 0
                    # print copy_components.counts[:copy_components.K]
                    # print loss_useful_cluster_num

                    for i in xrange(copy_components.N):

                        # Cache some old values for possible future use
                        k_old = copy_components.assignments[i]
                        K_old = copy_components.K
                        stats_old = copy_components.cache_component_stats(k_old)

                        # Remove data vector `X[i]` from its current component
                        copy_components.del_item(i)

                        # Compute log probability of `X[i]` belonging to each component
                        log_prob_z = np.zeros(copy_components.K + 1, np.float)

                        log_prob_z[:copy_components.K] = np.log(copy_components.counts[:copy_components.K])
                        # (25.33) in Murphy, p. 886
                        log_prob_z[:copy_components.K] += copy_components.log_post_pred(i)
                        # Add one component to which nothing has been assigned
                        log_prob_z[-1] = math.log(self.kalpha) + copy_components.cached_log_prior[i]
                        prob_z = np.exp(log_prob_z - logsumexp(log_prob_z))

                        # Sample the new component assignment for `X[i]`
                        k = utils.draw_rand(prob_z)

                        loss_loop_data_cnt = 0
                        if k_old in loss_useful_cluster_num:
                            k = k_old
                        else:
                            while k not in loss_useful_cluster_num:
                                loss_loop_data_cnt += 1
                                if loss_loop_data_cnt >= 100:
                                    break
                                # print '2: {}'.format(k)
                                k = utils.draw_rand(prob_z)

                        # Add data item X[i] into its component `k`
                        if k == k_old and copy_components.K == K_old:
                            # Assignment same and no components have been removed
                            copy_components.restore_component_from_stats(k_old, *stats_old)
                            copy_components.assignments[i] = k_old
                        else:
                            # Add data item X[i] into its new component `k`
                            copy_components.add_item(i, k)

                    if flag_marg:
                        log_prob = self.log_marg_for_copy(copy_components)
                        if log_prob > max_prob:
                            max_prob = log_prob
                            max_prob_components = copy_components
                    else:
                        loss_local = utils.cluster_loss_inertia(copy_components.X, copy_components.assignments)

                        if loss_local < min_loss:
                            min_loss = loss_local
                            min_loss_components = copy_components

            ## compute ada_power number
            if flag_adapower and i_iter > adapower_burnin:
                adapower_thres = self.components.N * adapower_perct
                adapower_nk = self.components.counts[:self.components.K]
                small_perct = len(adapower_nk[np.where(adapower_nk <= adapower_thres)[0]]) * 1.0 / len(
                    adapower_nk)
                ada_power = 1.0 + (r_up - 1.0) * small_perct
                if i_iter % 20 == 0:
                    logging.info('Ada-power: {}'.format(ada_power))





            ## Loop over data items
            for i in xrange(self.components.N):

                # Cache some old values for possible future use
                k_old = self.components.assignments[i]
                K_old = self.components.K
                stats_old = self.components.cache_component_stats(k_old)

                # Remove data vector `X[i]` from its current component
                self.components.del_item(i)



                if flag_power:
                    ## for power
                    log_prob_z = (
                        np.ones(self.components.K_max) * np.log(
                            float(self.kalpha) / self.components.K_max + np.power(self.components.counts, n_power)
                        )
                    )
                elif flag_adapower and i_iter > adapower_burnin:
                    ## for ada-power
                    log_prob_z = (
                        np.ones(self.components.K_max) * np.log(
                            float(self.kalpha) / self.components.K_max + np.power(self.components.counts, ada_power)
                        )
                    )
                else:
                    # Compute log probability of `X[i]` belonging to each component
                    # (24.26) in Murphy, p. 843
                    log_prob_z = (
                        np.ones(self.components.K_max) * np.log(
                            float(self.kalpha) / self.components.K_max + self.components.counts
                        )
                    )
                # (24.23) in Murphy, p. 842
                log_prob_z[:self.components.K] += self.components.log_post_pred(i)
                # Empty (unactive) components
                log_prob_z[self.components.K:] += self.components.log_prior(i)
                prob_z = np.exp(log_prob_z - logsumexp(log_prob_z))

                # Sample the new component assignment for `X[i]`
                k = utils.draw(prob_z)

                if flag_constrain:
                    if isConstrained:
                        # logging.info('performing constrained reassign')
                        if k_old in tmp_nonuseful_cluster_num:
                            k = utils.draw(prob_z)
                            while k not in tmp_useful_cluster_num:
                                k = utils.draw(prob_z)
                        else:
                            k = k_old

                # There could be several empty, unactive components at the end
                if k > self.components.K:
                    k = self.components.K
                # print prob_z, k, prob_z[k]

                # Add data item X[i] into its component `k`
                if k == k_old and self.components.K == K_old:
                    # Assignment same and no components have been removed
                    self.components.restore_component_from_stats(k_old, *stats_old)
                    self.components.assignments[i] = k_old
                else:
                    # Add data item X[i] into its new component `k`
                    self.components.add_item(i, k)




            if flag_loss and i_iter % n_loss_step == 0 and i_iter > loss_burnin:
                if flag_marg:
                    self.components = max_prob_components
                else:
                    self.components = min_loss_components

            # Update record
            record_dict = self.update_record_dict(record_dict, i_iter, true_assignments, start_time)
            start_time = time.time()
            distribution_dict, weights = self.update_distribution_dict(distribution_dict, weight_first,
                                                                       i_iter=i_iter)

            ## update for hyper prior on concentration parameter alpha
            if self.hyperprior is not None:
                weight_prod = np.prod(weights)
                K = self.components.K
                ars = ARS(f_hyperprior_log, f_hyperprior_log_prima, xi=[0.1, 1, 10], lb=0,
                          weight_prod=weight_prod, K=K, a=self.a, b=self.b)
                if not ars.no:
                    self.alpha = ars.draw(3)[2]
                    self.kalpha = self.alpha * self.K
                else:
                    self.kalpha = self.kalpha
                # distribution_dict["alpha"][i_iter] = self.kalpha

            ## Log info
            info = "iteration: " + str(i_iter)
            for key in sorted(record_dict):
                info += ", " + key + ": " + str(record_dict[key][-1])
            info += "."

            logger.info(info)

        return record_dict, distribution_dict


