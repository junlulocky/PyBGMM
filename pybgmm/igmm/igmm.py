"""
Base class for Infinite Gaussian mixture model (IGMM)

Date: 2017
"""

from numpy.linalg import cholesky, det, inv, slogdet
from scipy.misc import logsumexp
from scipy.special import gammaln
import logging
import math
import numpy as np
import time
from scipy import stats
import copy
import matplotlib.pyplot as plt


from ..gaussian.gaussian_components import GaussianComponents
from ..gaussian.gaussian_components_diag import GaussianComponentsDiag
from ..gaussian.gaussian_components_fixedvar import GaussianComponentsFixedVar

from ..utils import utils
from ..utils.plot_utils import plot_ellipse, plot_mixture_model

from ..gmm import GMM



logger = logging.getLogger(__name__)



#-----------------------------------------------------------------------------#
#                                  IGMM CLASS                                 #
#-----------------------------------------------------------------------------#

class IGMM(GMM):
    """
    An infinite Gaussian mixture model (IGMM).

    See `GaussianComponents` for an overview of the parameters not mentioned
    below.

    Parameters
    ----------
    alpha : float
        Concentration parameter for the Dirichlet process.
    assignments : vector of int or str
        If vector of int, this gives the initial component assignments. The
        vector should therefore have N entries between 0 and `K`. Values of
        -1 is also allowed, indicating that the data vector does not belong to
        any component. Alternatively, `assignments` can take one of the
        following values:
        - "rand": Vectors are assigned randomly to one of `K` components.
        - "one-by-one": Vectors are assigned one at a time; the value of
          `K` becomes irrelevant.
        - "each-in-own": Each vector is assigned to a component of its own.
    K : int
        The initial number of mixture components: this is only used when
        `assignments` is "rand".
    covariance_type : str
        String describing the type of covariance parameters to use. Must be
        one of "full", "diag" or "fixed".
    """


    def __init__(
            self, X, kernel_prior, alpha, save_path, assignments="rand", K=1, K_max=None,
            covariance_type="full"
            ):
        super(IGMM, self).__init__()

        data_shape = X.shape
        if len(data_shape) < 2:
            raise ValueError('X must be at least a 2-dimensional array.')

        self.save_path = save_path

        self.alpha = alpha
        self.N, self.D = X.shape



        # Initial component assignments
        if assignments == "rand":
            assignments = np.random.randint(0, K, self.N)

            # Make sure we have consequetive values
            for k in xrange(assignments.max()):
                while len(np.nonzero(assignments == k)[0]) == 0:
                    assignments[np.where(assignments > k)] -= 1
                if assignments.max() == k:
                    break
        elif assignments == "one-by-one":
            assignments = -1*np.ones(self.N, dtype="int")
            assignments[0] = 0  # first data vector belongs to first component
        elif assignments == "each-in-own":
            assignments = np.arange(self.N)
        else:
            # assignments is a vector
            pass

        if covariance_type == "full":
            self.components = GaussianComponents(X, kernel_prior, assignments, K_max)
        elif covariance_type == "diag":
            self.components = GaussianComponentsDiag(X, kernel_prior, assignments, K_max)
        elif covariance_type == "fixed":
            self.components = GaussianComponentsFixedVar(X, kernel_prior, assignments, K_max)
        else:
            assert False, "Invalid covariance type."



    def setup_distribution_dict(self, num_saved):
        """
        setup the distribution dictionary, with the input of number to saved
        :param num_saved: number of clusters to save
        :return: distribution dictionary
        """
        distribution_dict = {}
        distribution_dict["mean"] = np.zeros(shape=(num_saved, 0))
        distribution_dict["variance"] = np.zeros(shape=(num_saved, 0))
        distribution_dict["weights"] = np.zeros(shape=(num_saved, 0))

        return distribution_dict

    def update_distribution_dict(self, distribution_dict, weight_first):
        """
        update distribution dicrionary
        :param distribution_dict: dictionbution dictionary
        :param weight_first: update by weight first or mean first
        :return: distribution dictionary
        """

        ### if dimension = 2; save plot for ellipses in the save_path
        if self.D == 2:
            ## Plot results
            fig = plt.figure()
            ax = fig.add_subplot(111)
            plot_mixture_model(ax, self)
            plt.savefig(self.save_path + '/model.png')
            plt.savefig(self.save_path + '/model.pdf')
            for k in xrange(self.components.K):
                mu, sigma = self.components.map(k)
                plot_ellipse(ax, mu, sigma)
            # plt.show()
            plt.savefig(self.save_path + '/ellipse.png')
            plt.savefig(self.save_path + '/ellipse.pdf')

        ## get mean, sd and weights
        means = []
        sds = []
        for k in xrange(self.components.K):
            mu, sigma = self.components.map(k)
            means.append(mu)
            sds.append(sigma)

        if weight_first:
            ## label switching index
            weights = self.gibbs_weight()
            idx = np.argsort(weights)

            sds = np.array(sds).flatten()
            means = np.array(means).flatten()
            # weights = self.gibbs_weight()

            # label switching
            means = self.label_switch(idx, means)
            sds = self.label_switch(idx, sds)
            weights = self.label_switch(idx, weights)
        else:
            ## label switching index
            means = np.array(means).flatten()
            idx = np.argsort(means)

            sds = np.array(sds).flatten()
            weights = self.gibbs_weight()

            # label switching
            means = self.label_switch(idx, means)
            sds = self.label_switch(idx, sds)
            weights = self.label_switch(idx, weights)

        # back up for next iteration
        self.old_mean = means
        self.old_sigma = sds

        means = means.reshape((means.shape[0], 1))
        sds = sds.reshape((sds.shape[0], 1))
        weights = weights.reshape((weights.shape[0], 1))
        distribution_dict["mean"] = np.hstack((distribution_dict["mean"], means))
        distribution_dict["variance"] = np.hstack((distribution_dict["variance"], sds))
        distribution_dict["weights"] = np.hstack((distribution_dict["weights"], weights))


        return distribution_dict

    def log_marg(self):
        """Return log marginal of data and component assignments: p(X, z)"""

        # Log probability of component assignment P(z|alpha)
        # Equation (10) in Wood and Black, 2008
        # Use \Gamma(n) = (n - 1)!
        facts_ = gammaln(self.components.counts[:self.components.K])
        facts_[self.components.counts[:self.components.K] == 0] = 0  # definition of log(0!)
        log_prob_z = (
            (self.components.K - 1)*math.log(self.alpha) + gammaln(self.alpha)
            - gammaln(np.sum(self.components.counts[:self.components.K])
            + self.alpha) + np.sum(facts_)
            )

        log_prob_X_given_z = self.components.log_marg()

        return log_prob_z + log_prob_X_given_z



    def gibbs_weight(self):
        """
        Get weight vector for each gibbs iteration
        :return: weight vector
        """
        Nk = self.components.counts[:self.components.K].tolist()
        alpha = [Nk[cid] + self.alpha / self.components.K
                 for cid in range(self.components.K)]
        return stats.dirichlet(alpha).rvs(size=1).flatten()

    # # @profile
    # def gibbs_sample(self, n_iter, _true_assignment, n_print=20):
    #     """
    #     Perform `n_iter` iterations Gibbs sampling on the IGMM.
    #
    #     A record dict is constructed over the iterations, which contains
    #     several fields describing the sampling process. Each field is described
    #     by its key and statistics are given in a list which covers the Gibbs
    #     sampling iterations. This dict is returned.
    #     """
    #
    #     # Setup record dictionary
    #     record_dict = {}
    #     record_dict["sample_time"] = []
    #     start_time = time.time()
    #     record_dict["log_marg"] = []
    #     record_dict["components"] = []
    #     record_dict["nmi"] = []
    #     record_dict["mi"] = []
    #     record_dict["nk"] = []
    #
    #     # Loop over iterations
    #     for i_iter in range(n_iter):
    #
    #         # Loop over data items
    #         # import random
    #         # permuted = range(self.components.N)
    #         # random.shuffle(permuted)
    #         # for i in permuted:
    #         for i in xrange(self.components.N):
    #
    #             # Cache some old values for possible future use
    #             k_old = self.components.assignments[i]
    #             K_old = self.components.K
    #             stats_old = self.components.cache_component_stats(k_old)
    #
    #             # Remove data vector `X[i]` from its current component
    #             self.components.del_item(i)
    #
    #             # Compute log probability of `X[i]` belonging to each component
    #             log_prob_z = np.zeros(self.components.K + 1, np.float)
    #             # (25.35) in Murphy, p. 886
    #             log_prob_z[:self.components.K] = np.log(self.components.counts[:self.components.K])
    #             # (25.33) in Murphy, p. 886
    #             log_prob_z[:self.components.K] += self.components.log_post_pred(i)
    #             # Add one component to which nothing has been assigned
    #             log_prob_z[-1] = math.log(self.alpha) + self.components.cached_log_prior[i]
    #             prob_z = np.exp(log_prob_z - logsumexp(log_prob_z))
    #
    #             # Sample the new component assignment for `X[i]`
    #             k = utils.draw(prob_z)
    #             # logger.debug("Sampled k = " + str(k) + " from " + str(prob_z) + ".")
    #
    #             # Add data item X[i] into its component `k`
    #             if k == k_old and self.components.K == K_old:
    #                 # Assignment same and no components have been removed
    #                 self.components.restore_component_from_stats(k_old, *stats_old)
    #                 self.components.assignments[i] = k_old
    #             else:
    #                 # Add data item X[i] into its new component `k`
    #                 self.components.add_item(i, k)
    #
    #         # Update record
    #         record_dict["sample_time"].append(time.time() - start_time)
    #         start_time = time.time()
    #         record_dict["log_marg"].append(self.log_marg())
    #         record_dict["components"].append(self.components.K)
    #         nmi = normalized_mutual_info_score(_true_assignment, self.components.assignments)
    #         record_dict["nmi"].append(nmi)
    #         mi = mutual_info_score(_true_assignment, self.components.assignments)
    #         record_dict["mi"].append(mi)
    #         record_dict["nk"].append(self.components.counts[:self.components.K])
    #
    #         # Log info
    #         info = "iteration: " + str(i_iter)
    #         for key in sorted(record_dict):
    #             info += ", " + key + ": " + str(record_dict[key][-1])
    #         # info += ", nmi: " + str(nmi)
    #         info += "."
    #         logger.info(info)
    #
    #     return record_dict






    # def approx_sampling(self, n_iter, _true_assignment, approx_thres_perct=0.04, approx_burnin=200, num_saved=3):
    #
    #     return self.constrained_gibbs_sample(n_iter, _true_assignment,
    #                                       flag_constrain=False, n_constrain=1000000, thres=0,
    #                                       flag_power=False, n_power=1, power_burnin=1000000,
    #                                       flag_loss=False, n_loss_step=1000000, flag_marg=True, loss_burnin=1000000,
    #                                       flag_approx=True, approx_thres_perct=approx_thres_perct,
    #                                          approx_burnin=approx_burnin,
    #                                       num_saved=num_saved,)
    #
    # def ada_pcrp_sampling(self, n_iter, _true_assignment, r_up=1.1, adapcrp_perct=0.04, adapcrp_burnin=500,
    #                       num_saved=3):
    #     return self.constrained_gibbs_sample(n_iter, _true_assignment,
    #                                       flag_constrain=False, n_constrain=1000000, thres=0,
    #                                       flag_power=False, n_power=1, power_burnin=1000000,
    #                                       flag_loss=False, n_loss_step=1000000, flag_marg=True, loss_burnin=1000000,
    #                                       flag_approx=False, approx_thres_perct=0,
    #                                          approx_burnin=1000000,
    #                                       flag_adapcrp=True, r_up=r_up, adapcrp_perct=adapcrp_perct,
    #                                          adapcrp_burnin=adapcrp_burnin,
    #                                       num_saved=num_saved,)
    #
    # def ada_pcrp_sampling_form2(self, n_iter, _true_assignment, r_up=1.1, adapcrp_perct=0.04, adapcrp_burnin=500,
    #                       num_saved=3):
    #     return self.constrained_gibbs_sample(n_iter, _true_assignment,
    #                                       flag_constrain=False, n_constrain=1000000, thres=0,
    #                                       flag_power=False, n_power=1, power_burnin=1000000,
    #                                       flag_loss=False, n_loss_step=1000000, flag_marg=True, loss_burnin=1000000,
    #                                       flag_approx=False, approx_thres_perct=0,
    #                                          approx_burnin=1000000,
    #                                       flag_adapcrp=False, r_up=r_up, adapcrp_perct=adapcrp_perct,
    #                                          adapcrp_burnin=adapcrp_burnin,
    #                                       flag_adapcrp_form2=True,
    #                                       num_saved=num_saved,)
    #
    # def loss_ada_pcrp_sampling(self, n_iter, _true_assignment, r_up=1.2, adapcrp_step=0.01, adapcrp_burnin=500,
    #                       num_saved=3):
    #     return self.constrained_gibbs_sample(n_iter, _true_assignment,
    #                                       flag_constrain=False, n_constrain=1000000, thres=0.,
    #                                       flag_power=False, n_power=1, power_burnin=100000,
    #                                       flag_loss=False, n_loss_step=1000000, flag_marg=True, loss_burnin=10000000,
    #                                       flag_approx=False, approx_thres_perct=0., approx_burnin=1000000,
    #                                       flag_adapcrp=False, r_up=1., adapcrp_perct=0., adapcrp_burnin=1000000,
    #                                       flag_adapcrp_form2=False,
    #                                          flag_loss_adapcrp=True, r_up_losspcrp=r_up, lossadapcrp_step=adapcrp_step,
    #                                          lossadapcrp_burnin=adapcrp_burnin,
    #                                       num_saved=num_saved, weight_first=True)

    # # @profile
    # def constrained_gibbs_sample(self, n_iter, true_assignments,
    #                                       flag_constrain=False, n_constrain=1000000, thres=0.,
    #                                       flag_power=False, n_power=1, power_burnin=100000,
    #                                       flag_loss=False, n_loss_step=1000000, flag_marg=True, loss_burnin=10000000,
    #                                       flag_approx=False, approx_thres_perct=0., approx_burnin=1000000,
    #                                       flag_adapcrp=False, r_up=1., adapcrp_perct=0., adapcrp_burnin=1000000,
    #                                       flag_adapcrp_form2=False,
    #                                       flag_loss_adapcrp=False, r_up_losspcrp=1., lossadapcrp_step=0.,
    #                                                         lossadapcrp_burnin=1000000,
    #                                       num_saved=3, weight_first=True):
    #     """
    #     Perform `n_iter` iterations Gibbs sampling on the IGMM.
    #
    #     A record dict is constructed over the iterations, which contains
    #     several fields describing the sampling process. Each field is described
    #     by its key and statistics are given in a list which covers the Gibbs
    #     sampling iterations. This dict is returned.
    #     """
    #
    #     # Setup record dictionary
    #     record_dict = self.setup_record_dict()
    #     start_time = time.time()
    #     distribution_dict = {}
    #
    #     # record_dict["sample_time"] = []
    #     # start_time = time.time()
    #     # record_dict["log_marg"] = []
    #     # record_dict["components"] = []
    #     # record_dict["nmi"] = []
    #     # record_dict["mi"] = []
    #     # record_dict["nk"] = []
    #     # record_dict["loss"] = []
    #     # record_dict['bic'] = []
    #     # record_dict["vi"] = []
    #
    #     distribution_dict["mean"] = np.zeros(shape=(num_saved, 0))
    #     distribution_dict["variance"] = np.zeros(shape=(num_saved, 0))
    #     distribution_dict["weights"] = np.zeros(shape=(num_saved, 0))
    #
    #     dist_idx = 0
    #
    #     constrain_thres = self.components.N * thres
    #
    #
    #
    #     if flag_loss_adapcrp:
    #         smallest_loss_adapcrp = utils.cluster_loss_inertia(self.components.X, self.components.assignments)
    #         r_lossadapcrp = 1.  ## initial power
    #
    #     all_noise_data = []
    #
    #     # Loop over iterations
    #     for i_iter in range(n_iter):
    #         # print 'iter: {}'.format(i_iter)
    #
    #         isNoiseAnalysis = False
    #         if isNoiseAnalysis:
    #             # logging.info('clusters:{}'.format(self.components.counts[:self.components.K]))
    #             small_cluster_idx = np.where(self.components.counts[:self.components.K]<=1)[0]
    #             # logging.info('less than 2:{}'.format(small_cluster_idx))
    #             # logging.info('assignments: {}'.format(collections.Counter(self.components.assignments)))
    #
    #             data_idx = [i for i,row in enumerate(self.components.assignments) if row in small_cluster_idx]
    #             logging.info("data idx:{}".format(data_idx))
    #
    #             all_noise_data = all_noise_data + data_idx
    #             logging.info("unique idx:{}".format(np.unique(all_noise_data)))
    #
    #
    #         ## save the wanted distribution
    #         if num_saved == self.components.K and i_iter>1:
    #
    #             ### dimension = 2; save plot
    #             if self.D == 2:
    #                 ## Plot results
    #                 fig = plt.figure()
    #                 ax = fig.add_subplot(111)
    #                 plot_mixture_model(ax, self)
    #                 plt.savefig(self.save_path + '/model.png')
    #                 plt.savefig(self.save_path + '/model.pdf')
    #                 for k in xrange(self.components.K):
    #                     mu, sigma = self.components.map(k)
    #                     plot_ellipse(ax, mu, sigma)
    #                 # plt.show()
    #                 plt.savefig(self.save_path + '/ellipse.png')
    #                 plt.savefig(self.save_path + '/ellipse.pdf')
    #
    #             ## get mean, sd and weights
    #             means = []
    #             sds = []
    #             for k in xrange(self.components.K):
    #                 mu, sigma = self.components.map(k)
    #                 means.append(mu)
    #                 sds.append(sigma)
    #
    #             if weight_first:
    #                 ## label switching index
    #                 weights = self.gibbs_weight()
    #                 idx = np.argsort(weights)
    #
    #                 sds = np.array(sds).flatten()
    #                 means = np.array(means).flatten()
    #                 # weights = self.gibbs_weight()
    #
    #                 # label switching
    #                 means = self.label_switch(idx, means)
    #                 sds = self.label_switch(idx, sds)
    #                 weights = self.label_switch(idx, weights)
    #             else:
    #                 ## label switching index
    #                 means = np.array(means).flatten()
    #                 idx = np.argsort(means)
    #
    #                 sds = np.array(sds).flatten()
    #                 weights = self.gibbs_weight()
    #
    #                 # label switching
    #                 means = self.label_switch(idx, means)
    #                 sds = self.label_switch(idx, sds)
    #                 weights = self.label_switch(idx, weights)
    #
    #             # back up for next iteration
    #             self.old_mean = means
    #             self.old_sigma = sds
    #
    #             means = means.reshape((means.shape[0], 1))
    #             sds = sds.reshape((sds.shape[0],1))
    #             weights = weights.reshape((weights.shape[0],1))
    #             distribution_dict["mean"] = np.hstack((distribution_dict["mean"], means))
    #             distribution_dict["variance"] = np.hstack((distribution_dict["variance"], sds))
    #             distribution_dict["weights"] = np.hstack((distribution_dict["weights"], weights))
    #
    #             dist_idx = dist_idx + 1
    #
    #
    #
    #
    #         if flag_constrain:
    #             if i_iter % n_constrain == 0:
    #                 logging.info('performing constrain step')
    #                 logging.info('all cluster nk: {}'.format(self.components.counts[:self.components.K]))
    #                 isConstrained = True
    #                 tmp_useful_cluster_num = []
    #                 tmp_nonuseful_cluster_num = []
    #                 for i_cluster in range(self.components.K):
    #                     if self.components.counts[i_cluster] > constrain_thres:
    #                         tmp_useful_cluster_num.append(i_cluster)
    #                     else:
    #                         tmp_nonuseful_cluster_num.append(i_cluster)
    #             else:
    #                 isConstrained = False
    #             # print self.components.K
    #             # print self.components.counts
    #
    #         if flag_loss and i_iter % n_loss_step == 0 and i_iter > loss_burnin:
    #             copy_components = copy.deepcopy(self.components)
    #             if flag_marg:
    #                 max_prob = float('-inf')
    #                 max_prob_components = copy_components
    #             else:
    #                 min_loss = float('+inf')
    #                 min_loss_components = copy_components
    #
    #             loss_cnt=0
    #             while copy_components.K > 2:
    #                 loss_cnt += 1
    #                 if loss_cnt > 50:
    #                     break
    #                 # print "iter: {}".format(i_iter)
    #                 # print "1: {}".format(copy_components.K)
    #
    #                 # because we need to assign the copy to max_components
    #                 copy_components = copy.deepcopy(copy_components)
    #
    #                 loss_nonuseful_cluster_idx = np.argmin(copy_components.counts[:copy_components.K])
    #                 loss_useful_cluster_num = []
    #
    #                 for i_cluster in range(copy_components.K):
    #                     if i_cluster != loss_nonuseful_cluster_idx:
    #                         loss_useful_cluster_num.append(i_cluster)
    #                 # tmp_counts = copy_components.counts[:copy_components.K]
    #                 # tmp_counts[loss_nonuseful_cluster_idx] = 0
    #                 # print copy_components.counts[:copy_components.K]
    #                 # print loss_useful_cluster_num
    #
    #                 for i in xrange(copy_components.N):
    #
    #                     # Cache some old values for possible future use
    #                     k_old = copy_components.assignments[i]
    #                     K_old = copy_components.K
    #                     stats_old = copy_components.cache_component_stats(k_old)
    #
    #                     # Remove data vector `X[i]` from its current component
    #                     copy_components.del_item(i)
    #
    #                     # Compute log probability of `X[i]` belonging to each component
    #                     log_prob_z = np.zeros(copy_components.K + 1, np.float)
    #
    #                     log_prob_z[:copy_components.K] = np.log(copy_components.counts[:copy_components.K])
    #                     # (25.33) in Murphy, p. 886
    #                     log_prob_z[:copy_components.K] += copy_components.log_post_pred(i)
    #                     # Add one component to which nothing has been assigned
    #                     log_prob_z[-1] = math.log(self.alpha) + copy_components.cached_log_prior[i]
    #                     prob_z = np.exp(log_prob_z - logsumexp(log_prob_z))
    #
    #                     # Sample the new component assignment for `X[i]`
    #                     k = utils.draw_rand(prob_z)
    #
    #                     loss_loop_data_cnt = 0
    #                     if k_old in loss_useful_cluster_num:
    #                         k = k_old
    #                     else:
    #                         while k not in loss_useful_cluster_num:
    #                             loss_loop_data_cnt += 1
    #                             if loss_loop_data_cnt >= 100:
    #                                 break
    #                             # print '2: {}'.format(k)
    #                             k = utils.draw_rand(prob_z)
    #
    #                     # Add data item X[i] into its component `k`
    #                     if k == k_old and copy_components.K == K_old:
    #                         # Assignment same and no components have been removed
    #                         copy_components.restore_component_from_stats(k_old, *stats_old)
    #                         copy_components.assignments[i] = k_old
    #                     else:
    #                         # Add data item X[i] into its new component `k`
    #                         copy_components.add_item(i, k)
    #
    #                     ## TODO: move out
    #                     if flag_marg:
    #                         log_prob = self.log_marg_for_copy(copy_components)
    #                         if log_prob > max_prob:
    #                             max_prob = log_prob
    #                             max_prob_components = copy_components
    #                     else:
    #                         loss_local = utils.cluster_loss_inertia(copy_components.X, copy_components.assignments)
    #                         if loss_local < min_loss:
    #                             min_loss = loss_local
    #                             min_loss_components = copy_components
    #
    #         if flag_adapcrp_form2 and i_iter > adapcrp_burnin:  # for ada-pCRP
    #             adapcrp_thres = self.components.N * adapcrp_perct
    #             adapcrp_nk = self.components.counts[:self.components.K]
    #             small_perct = len(adapcrp_nk[np.where(adapcrp_nk <= adapcrp_thres)[0]]) * 1.0 / len(adapcrp_nk)
    #             adapcrp_power_form2 = 1.0 + (r_up - 1.0) * small_perct
    #             if i_iter % 20 == 0:
    #                 logging.info('Ada-pCRP power: {}'.format(adapcrp_power_form2))
    #
    #         ## parameter prepare for 'loss_adapcrp'
    #         if flag_loss_adapcrp and i_iter > lossadapcrp_burnin:
    #             this_loss = utils.cluster_loss_inertia(self.components.X, self.components.assignments)
    #             if this_loss < smallest_loss_adapcrp:
    #                 r_lossadapcrp -= lossadapcrp_step
    #                 smallest_loss_adapcrp = this_loss
    #             else:
    #                 r_lossadapcrp += lossadapcrp_step
    #
    #             if r_lossadapcrp < 1.:
    #                 r_lossadapcrp = 1.
    #             if r_lossadapcrp > r_up_losspcrp:
    #                 r_lossadapcrp = r_up_losspcrp
    #
    #
    #             if i_iter % 20 == 0:
    #                 logging.info('smallest loss: {}'.format(smallest_loss_adapcrp))
    #                 logging.info('loss: {}'.format(this_loss))
    #                 logging.info('power: {}'.format(r_lossadapcrp))
    #
    #         if flag_power and n_power>1:
    #             if i_iter % 20 == 0:
    #                 print "permutate data"
    #             data_loop_list = np.random.permutation(xrange(self.components.N))
    #         else:
    #             data_loop_list = xrange(self.components.N)
    #         ## Loop over data items
    #         for i in data_loop_list:
    #
    #             # Cache some old values for possible future use
    #             k_old = self.components.assignments[i]
    #             K_old = self.components.K
    #             stats_old = self.components.cache_component_stats(k_old)
    #
    #             # Remove data vector `X[i]` from its current component
    #             self.components.del_item(i)
    #
    #             # Compute log probability of `X[i]` belonging to each component
    #             log_prob_z = np.zeros(self.components.K + 1, np.float)
    #             if flag_power and i_iter > power_burnin:
    #                 ## for pCRP
    #                 log_prob_z[:self.components.K] = np.log(np.power(self.components.counts[:self.components.K],n_power))
    #             elif flag_adapcrp and i_iter > adapcrp_burnin:
    #                 ## for ada-pCRP
    #                 adapcrp_thres = self.components.N * adapcrp_perct
    #                 adapcrp_nk = self.components.counts[:self.components.K]
    #                 small_perct = len(adapcrp_nk[np.where(adapcrp_nk<=adapcrp_thres)[0]]) * 1.0 /len(adapcrp_nk)
    #                 adapcrp_power = 1.0 + (r_up-1.0)*small_perct
    #                 # logging.info('Ada-pCRP power: {}'.format(adapcrp_power))
    #                 log_prob_z[:self.components.K] = np.log(
    #                     np.power(self.components.counts[:self.components.K], adapcrp_power))
    #             elif flag_adapcrp_form2 and i_iter > adapcrp_burnin:
    #                 ## for ada-pCRP form2
    #                 log_prob_z[:self.components.K] = np.log(
    #                     np.power(self.components.counts[:self.components.K], adapcrp_power_form2))
    #             elif flag_loss_adapcrp and i_iter > lossadapcrp_burnin:
    #                 ## for loss-ada-pCRP
    #                 log_prob_z[:self.components.K] = np.log(
    #                     np.power(self.components.counts[:self.components.K], r_lossadapcrp))
    #             else:
    #                 ## plain gibbs sampling
    #                 # (25.35) in Murphy, p. 886
    #                 log_prob_z[:self.components.K] = np.log(self.components.counts[:self.components.K])
    #             # (25.33) in Murphy, p. 886
    #             log_prob_z[:self.components.K] += self.components.log_post_pred(i)
    #             # Add one component to which nothing has been assigned
    #             log_prob_z[-1] = math.log(self.alpha) + self.components.cached_log_prior[i]
    #             prob_z = np.exp(log_prob_z - logsumexp(log_prob_z))
    #
    #             # Sample the new component assignment for `X[i]`
    #             k = utils.draw(prob_z)
    #             # logger.debug("Sampled k = " + str(k) + " from " + str(prob_z) + ".")
    #
    #             if flag_constrain:
    #                 if isConstrained:
    #                     # logging.info('performing constrained reassign')
    #                     if k_old in tmp_nonuseful_cluster_num:
    #                         k = utils.draw(prob_z)
    #                         while k not in tmp_useful_cluster_num:
    #                             k = utils.draw(prob_z)
    #                     else:
    #                         k = k_old
    #
    #             # Add data item X[i] into its component `k`
    #             if k == k_old and self.components.K == K_old:
    #                 # Assignment same and no components have been removed
    #                 self.components.restore_component_from_stats(k_old, *stats_old)
    #                 self.components.assignments[i] = k_old
    #             else:
    #                 # Add data item X[i] into its new component `k`
    #                 self.components.add_item(i, k)
    #         ## end loop data
    #
    #         ## noise proof
    #         isNoiseProof = False
    #         if isNoiseProof:
    #             noise_useful_cluster_num = []
    #             noise_nonuseful_cluster_num = []
    #             for i_cluster in range(self.components.K):
    #                 if self.components.counts[i_cluster] == 1:
    #                     noise_nonuseful_cluster_num.append(i_cluster)
    #                 else:
    #                     noise_useful_cluster_num.append(i_cluster)
    #
    #             # logging.info('clusters:{}'.format(self.components.counts[:self.components.K]))
    #             small_cluster_idx = np.where(self.components.counts[:self.components.K] == 1)[0]
    #
    #
    #             small_data_idx = [i for i, row in enumerate(self.components.assignments) if row in small_cluster_idx]
    #
    #             ## Loop over data items
    #             for i in small_data_idx:
    #
    #                 # Cache some old values for possible future use
    #                 k_old = self.components.assignments[i]
    #                 K_old = self.components.K
    #                 stats_old = self.components.cache_component_stats(k_old)
    #
    #                 # Remove data vector `X[i]` from its current component
    #                 self.components.del_item(i)
    #
    #                 # Compute log probability of `X[i]` belonging to each component
    #                 log_prob_z = np.zeros(self.components.K + 1, np.float)
    #
    #                 ## plain gibbs sampling
    #                 # (25.35) in Murphy, p. 886
    #                 log_prob_z[:self.components.K] = np.log(self.components.counts[:self.components.K])
    #                 # (25.33) in Murphy, p. 886
    #                 log_prob_z[:self.components.K] += self.components.log_post_pred(i)
    #                 # Add one component to which nothing has been assigned
    #                 log_prob_z[-1] = math.log(self.alpha) + self.components.cached_log_prior[i]
    #                 prob_z = np.exp(log_prob_z - logsumexp(log_prob_z))
    #
    #                 k = utils.draw(prob_z)
    #                 while k in small_cluster_idx:
    #                     k = utils.draw(prob_z)
    #
    #
    #                 # Add data item X[i] into its component `k`
    #                 if k == k_old and self.components.K == K_old:
    #                     # Assignment same and no components have been removed
    #                     self.components.restore_component_from_stats(k_old, *stats_old)
    #                     self.components.assignments[i] = k_old
    #                 else:
    #                     # Add data item X[i] into its new component `k`
    #                     self.components.add_item(i, k)
    #         ## end noise proof
    #
    #
    #         if flag_approx and i_iter > approx_burnin:
    #             approx_thres = self.components.N * approx_thres_perct
    #             if i_iter % 20 == 0:
    #                 logging.info('performing approx step')
    #                 logging.info('all cluster nk: {}'.format(self.components.counts[:self.components.K]))
    #             approx_useful_cluster_num = []
    #             approx_nonuseful_cluster_num = []
    #             for i_cluster in range(self.components.K):
    #                 if self.components.counts[i_cluster] > approx_thres:
    #                     approx_useful_cluster_num.append(i_cluster)
    #                 else:
    #                     approx_nonuseful_cluster_num.append(i_cluster)
    #
    #             # Loop over data items
    #             for i in xrange(self.components.N):
    #
    #                 # Cache some old values for possible future use
    #                 k_old = self.components.assignments[i]
    #                 K_old = self.components.K
    #                 stats_old = self.components.cache_component_stats(k_old)
    #
    #                 # Remove data vector `X[i]` from its current component
    #                 self.components.del_item(i)
    #
    #                 # Compute log probability of `X[i]` belonging to each component
    #                 log_prob_z = np.zeros(self.components.K + 1, np.float)
    #                 # (25.35) in Murphy, p. 886
    #                 log_prob_z[:self.components.K] = np.log(self.components.counts[:self.components.K])
    #                 # (25.33) in Murphy, p. 886
    #                 log_prob_z[:self.components.K] += self.components.log_post_pred(i)
    #                 # Add one component to which nothing has been assigned
    #                 log_prob_z[-1] = math.log(self.alpha) + self.components.cached_log_prior[i]
    #                 prob_z = np.exp(log_prob_z - logsumexp(log_prob_z))
    #
    #                 # Sample the new component assignment for `X[i]`
    #                 k = utils.draw(prob_z)
    #
    #                 ## approx sampling step
    #                 if k_old in approx_nonuseful_cluster_num:
    #                     k = utils.draw(prob_z)
    #                     while k not in approx_useful_cluster_num:
    #                         k = utils.draw(prob_z)
    #                 else:
    #                     k = k_old
    #
    #                 # Add data item X[i] into its component `k`
    #                 if k == k_old and self.components.K == K_old:
    #                     # Assignment same and no components have been removed
    #                     self.components.restore_component_from_stats(k_old, *stats_old)
    #                     self.components.assignments[i] = k_old
    #                 else:
    #                     # Add data item X[i] into its new component `k`
    #                     self.components.add_item(i, k)
    #
    #
    #
    #         if flag_loss and i_iter % n_loss_step == 0 and i_iter>loss_burnin:
    #             if flag_marg:
    #                 self.components = max_prob_components
    #             else:
    #                 self.components = min_loss_components
    #
    #         # Update record
    #         record_dict = self.update_record_dict(record_dict, i_iter, true_assignments, start_time)
    #         start_time = time.time()
    #
    #         # record_dict["sample_time"].append(time.time() - start_time)
    #         # start_time = time.time()
    #         # record_dict["log_marg"].append(self.log_marg())
    #         # record_dict["components"].append(self.components.K)
    #         # nmi = normalized_mutual_info_score(_true_assignment, self.components.assignments)
    #         # record_dict["nmi"].append(nmi)
    #         # mi = mutual_info_score(_true_assignment, self.components.assignments)
    #         # record_dict["mi"].append(mi)
    #         # record_dict["nk"].append(self.components.counts[:self.components.K])
    #         # loss = utils.cluster_loss_inertia(self.components.X, self.components.assignments)
    #         # record_dict["loss"].append(loss)
    #         #
    #         # bic = utils.cluster_loss_inertia(self.components.X, self.components.assignments)
    #         # record_dict["bic"].append(bic)
    #         #
    #         # vi = information_variation(_true_assignment, self.components.assignments)
    #         # record_dict["vi"].append(vi)
    #         # Log info
    #         # if i_iter % 20 ==0:
    #         #     info = "iteration: " + str(i_iter)
    #         #     for key in sorted(record_dict):
    #         #         info += ", " + key + ": " + str(record_dict[key][-1])
    #         #     # info += ", nmi: " + str(nmi)
    #         #     info += "."
    #         #     logger.info(info)
    #
    #     return record_dict, distribution_dict





