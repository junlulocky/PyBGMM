"""
Gibbs sampler for Sub clustering by Chinese restaurant process mixture model (SubCRPMM)

Date: 2017
"""

from scipy.misc import logsumexp
import numpy as np
import time
import math
import logging
from numpy import linalg as LA
from scipy.special import gammaln
from scipy.stats import bernoulli
import copy


from .igmm import IGMM
from ..utils import utils

from ..gaussian.gaussian_components import GaussianComponents
from ..gaussian.gaussian_components_diag import GaussianComponentsDiag
from ..gaussian.gaussian_components_fixedvar import GaussianComponentsFixedVar

from ..prior.niw import NIW
from ..prior.betabern import BetaBern

logger = logging.getLogger(__name__)

class SubCRPMM(IGMM):

    def __init__(self, X, kernel_prior, alpha, save_path, assignments="rand", K=1, K_max=None,
            covariance_type="full", common_component_covariance_type="full", bern_prior=BetaBern(1,1), p_bern=0.1):
        super(SubCRPMM, self).__init__(X, kernel_prior, alpha, save_path, assignments=assignments, K=K, K_max=K_max,
            covariance_type=covariance_type)

        ## save for SubCRPMM
        self.X = X
        self.covariance_type = covariance_type
        self.common_component_covariance_type = common_component_covariance_type
        self.K_max = K_max

        self.h1 = 40 ## prior info on the covariance matrix for CRP
        self.h0 = 40 ## prior info on the covariance matrix for common component

        if bern_prior is not None:
            self.bern_prior = bern_prior
            self.p_bern = 1. * bern_prior.a / (bern_prior.a + bern_prior.b)
            self.make_robust_p_bern()
        else:
            self.bern_prior = None
            self.p_bern = p_bern
            self.make_robust_p_bern()

        ## count acceptance rate of metropolis search
        self.total_run = 0
        self.update_run = 0
        self.acc_rate = 1


        ## initial mask by bernoulli variable - now make all to 1, used for SubCRP-burnin
        # self.mask = np.random.binomial(1, self.p_bern, self.D)
        self.mask = np.random.binomial(1, 1, self.D)

        """STEP 1 for common component"""
        self.common_component = self.update_common_component(1-self.mask)

        """STEP 2 for cluster components"""
        cluster_idx = 1
        cluster_X = self.X[:, np.where(self.mask == cluster_idx)[0]]
        # Initial component assignments
        if assignments == "rand":
            assignments = np.random.randint(0, K, cluster_X.shape[0])

            # Make sure we have consequetive values
            for k in xrange(assignments.max()):
                while len(np.nonzero(assignments == k)[0]) == 0:
                    assignments[np.where(assignments > k)] -= 1
                if assignments.max() == k:
                    break
        elif assignments == "one-by-one":
            assignments = -1 * np.ones(cluster_X.shape[0], dtype="int")
            assignments[0] = 0  # first data vector belongs to first component
        elif assignments == "each-in-own":
            assignments = np.arange(cluster_X.shape[0])
        else:
            # assignments is a vector
            pass
        self.components = self.update_clustering_components(self.mask, assignments)


    def make_robust_p_bern(self):
        """
        make the probability of bernoulli robust, 1 or 0 will get -inf
        :return: robust version of p_bern
        """
        isRobustBern = True
        if isRobustBern:
            if self.p_bern == 1.0:
                self.p_bern = 1.0 - 0.0001
            if self.p_bern == 0.0:
                self.p_bern = 0.0001


    def log_marg_for_specific_component(self, components):
        """Return log marginal of data and component assignments: p(X, z)"""

        # Log probability of component assignment P(z|alpha)
        # Equation (10) in Wood and Black, 2008
        # Use \Gamma(n) = (n - 1)!
        facts_ = gammaln(components.counts[:components.K])
        facts_[self.components.counts[:components.K] == 0] = 0  # definition of log(0!)
        log_prob_z = (
            (components.K - 1) * math.log(self.alpha) + gammaln(self.alpha)
            - gammaln(np.sum(components.counts[:components.K])
                      + self.alpha) + np.sum(facts_)
        )

        log_prob_X_given_z = components.log_marg()

        return log_prob_z + log_prob_X_given_z

    def update_common_component(self, mask):
        common_idx = 0
        common_D = np.where(mask == common_idx)[0].shape[0]
        if np.sum(common_D) == 0:
            mask[-1] = 0
            common_D = np.where(mask == common_idx)[0].shape[0]
        common_X = self.X[:, np.where(mask == common_idx)[0]]

        if common_D == 1:
            covar_scale = np.var(common_X)
        else:
            covar_scale = np.median(LA.eigvals(np.cov(common_X.T)))
            # pass
        mu_scale = np.amax(common_X) - covar_scale

        m_0 = common_X.mean(axis=0)
        k_0 = 1.0 / self.h0
        # k_0 = covar_scale**2/mu_scale**2
        v_0 = common_D + 2
        # S_0 = 1. / covar_scale * np.eye(common_D)
        S_0 = 1. * np.eye(common_D)
        common_kernel_prior = NIW(m_0, k_0, v_0, S_0)

        ## save for common component, unused dimensions
        common_assignments = np.zeros(common_X.shape[0])  ## one component

        if self.common_component_covariance_type == "full":
            common_component = GaussianComponents(common_X, common_kernel_prior, common_assignments, 1)
        elif self.common_component_covariance_type == "diag":
            common_component = GaussianComponentsDiag(common_X, common_kernel_prior, common_assignments, 1)
        elif self.common_component_covariance_type == "fixed":
            common_component = GaussianComponentsFixedVar(common_X, common_kernel_prior, common_assignments, 1)
        else:
            assert False, "Invalid covariance type."

        return common_component

    def update_clustering_components(self, mask, assignments):
        cluster_idx = 1
        cluster_D = np.where(mask == cluster_idx)[0].shape[0]
        cluster_X = self.X[:, np.where(mask == cluster_idx)[0]]

        if cluster_D == 1:
            covar_scale = np.var(cluster_X)
        else:
            covar_scale = np.median(LA.eigvals(np.cov(cluster_X.T)))
        mu_scale = np.amax(cluster_X) - covar_scale

        # Intialize prior
        m_0 = cluster_X.mean(axis=0)
        k_0 = 1.0 / self.h1
        # k_0 = covar_scale ** 2 / mu_scale ** 2
        v_0 = cluster_D + 2
        # S_0 = 1./100 / covar_scale * np.eye(cluster_D)
        S_0 = 1. * np.eye(cluster_D)

        cluster_kernel_prior = NIW(m_0, k_0, v_0, S_0)

        if self.covariance_type == "full":
            components = GaussianComponents(cluster_X, cluster_kernel_prior, assignments, self.K_max)
        elif self.covariance_type == "diag":
            components = GaussianComponentsDiag(cluster_X, cluster_kernel_prior, assignments, self.K_max)
        elif self.covariance_type == "fixed":
            components = GaussianComponentsFixedVar(cluster_X, cluster_kernel_prior, assignments, self.K_max)
        else:
            assert False, "Invalid covariance type."

        return components

    def metropolis_update_mask(self, i_iter):

        assert self.common_component.K==1, "common component can only have one cluster component"
        ## compute old p(mask | z,X) \propto p(X,z|mask) * p(mask | p_bern)
        ## p(X,z | mask) = p(X_m, z | \alpha, \beta_m) * p(X_mc | \beta_mc)
        log_bern_old = np.sum(bernoulli.logpmf(self.mask, self.p_bern))  # log prob for bernoulli distribution
        log_marg_old = self.log_marg() + self.common_component.log_marg() + log_bern_old

        ## random pick one mask from old mask
        idx = np.random.choice(xrange(self.D),1)[0]
        mask_new = copy.deepcopy(self.mask)
        mask_new[idx] = 1 - mask_new[idx]

        ## robust step fpr new mask
        cluster_idx = 1
        cluster_D = np.where(mask_new == cluster_idx)[0].shape[0]
        common_idx = 0
        common_D = np.where(mask_new == common_idx)[0].shape[0]
        if cluster_D==0 or common_D ==0:
            logging.info('use old !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
            mask_new = copy.deepcopy(self.mask)

        ## TODO, test to see if mask are same, do we get same log marg
        # mask_new = copy.deepcopy(self.mask)

        # if i_iter % 20 == 0:
        #     logging.info('mask old: {}'.format(self.mask))
        #     logging.info('mask new: {}'.format(mask_new))

        ## compute new p(mask | z,X) \propto p(X,z|mask) * p(mask | p_bern)
        ## p(X,z | mask) = p(X_m, z | \alpha, \beta_m) * p(X_mc | \beta_mc)
        log_bern_new = np.sum(bernoulli.logpmf(mask_new, self.p_bern)) # log prob for bernoulli distribution

        # get common components and clustering components from new mask
        local_common_component = self.update_common_component(mask_new)
        assert self.common_component.K == 1, "new common component can only have one cluster component"
        local_clustering_components = self.update_clustering_components(mask_new, self.components.assignments)

        log_marg_new = self.log_marg_for_specific_component(local_clustering_components) + \
                       local_common_component.log_marg() + log_bern_new

        # if i_iter % 20 == 0:
        #     print 'log_bern_old: {}'.format(log_bern_old)
        #     print 'log_bern_new: {}'.format(log_bern_new)
        #
        #     print 'common component: {}'.format(self.common_component.log_marg())
        #     print 'local common component: {}'.format(local_common_component.log_marg())
        #
        #     print 'cluster component: {}'.format(self.components.log_marg())
        #     print 'local cluster component: {}'.format(local_clustering_components.log_marg())
        #
        #     print 'cluster component all: {}'.format(self.log_marg())
        #     print 'local cluster component all: {}'.format(self.log_marg_for_specific_component(
        #         local_clustering_components))


        isNewLarger = True
        self.total_run += 1  ## count run
        if isNewLarger:
            ## metropolis search,
            prob = np.exp(log_marg_new - log_marg_old)
            prob = 1 if prob > 1 else prob
            bern_prob = np.random.binomial(1, prob, 1)[0]

            # if i_iter % 20 == 0:
            #     print 'prob: {}'.format(prob)
            #     print 'prob output: {}'.format(bern_prob)

            if log_marg_new > log_marg_old or bern_prob > 0:
                self.update_run += 1  ## count update
                self.acc_rate = self.update_run * 1. / self.total_run  ## update acceptance rate
                if i_iter % 20 == 0:
                    logging.info('update mask!!')
                self.mask = copy.deepcopy(mask_new)
                self.common_component = copy.deepcopy(local_common_component)
                self.components = copy.deepcopy(local_clustering_components)

                ## update p_bern for the mask if using BetaBern prior
                if self.bern_prior is not None:
                    self.p_bern = np.random.beta(self.bern_prior.a + np.sum(mask_new),
                                                self.bern_prior.b + self.D - np.sum(mask_new), 1)[0]
                    self.make_robust_p_bern()
                else:
                    ## self.p_bern is fixed
                    pass
        else:
            ## metropolis search,
            prob = np.exp(log_marg_old - log_marg_new)
            prob = 1 if prob > 1 else prob
            bern_prob = np.random.binomial(1, prob, 1)[0]

            # if i_iter % 20 == 0:
            #     print 'prob: {}'.format(prob)
            #     print 'prob output: {}'.format(bern_prob)

            if log_marg_old > log_marg_new or bern_prob > 0:
                if i_iter % 20 == 0:
                    logging.info('update mask!!')
                self.mask = copy.deepcopy(mask_new)
                self.common_component = copy.deepcopy(local_common_component)
                self.components = copy.deepcopy(local_clustering_components)

                ## update p_bern for the mask if using BetaBern prior
                if self.bern_prior is not None:
                    self.p_bern = np.random.beta(self.bern_prior.a + np.sum(mask_new),
                                                 self.bern_prior.b + self.D - np.sum(mask_new), 1)[0]
                    self.make_robust_p_bern()

        if i_iter % 20 == 0:
            logging.info('log_marg_old: {}'.format(log_marg_old))
            logging.info('log_marg_new: {}'.format(log_marg_new))
            logging.info('p_bern_new: {}'.format(self.p_bern))
            logging.info('accetance rate: {}'.format(self.acc_rate))

    def log_marg_mask(self, mask_new):
        ## log prob for bernoulli distribution
        log_bern_new = np.sum(bernoulli.logpmf(mask_new, self.p_bern))

        # get common components and clustering components from new mask
        local_common_component = self.update_common_component(mask_new)
        assert self.common_component.K == 1, "new common component can only have one cluster component"
        local_clustering_components = self.update_clustering_components(mask_new, self.components.assignments)

        ## compute log prob for p(m) * p(X,z | m)
        log_marg_new = self.log_marg_for_specific_component(local_clustering_components) + \
                       local_common_component.log_marg() + log_bern_new

        return log_marg_new

    def gibbs_update_mask(self, i_iter):
        mask_old = copy.deepcopy(self.mask)
        assert self.common_component.K==1, "common component can only have one cluster component"

        for i_dim in range(self.D):
            mask_new = copy.deepcopy(self.mask)

            # Compute log probability of `mask[i]`
            log_prob_mask = np.zeros(2, np.float)
            mask_new[i_dim] = 0
            log_prob_mask[0] = self.log_marg_mask(mask_new)
            mask_new[i_dim] = 1
            log_prob_mask[1] = self.log_marg_mask(mask_new)


            prob_mask = np.exp(log_prob_mask - logsumexp(log_prob_mask))

            # Sample the new component assignment for `mask[i]`
            k = utils.draw(prob_mask)
            self.mask[i_dim] = k

        self.p_bern = np.random.beta(self.bern_prior.a + np.sum(self.mask),
                                     self.bern_prior.b + self.D - np.sum(self.mask), 1)[0]
        self.make_robust_p_bern()

        if i_iter % 20 == 0:
            logging.info('p_bern_new: {}'.format(self.p_bern))
            logging.info('mask old: {}'.format(mask_old))
            logging.info('mask new: {}'.format(self.mask))

    def setup_subcrp_record(self):
        subcrp_record_dict = {}

        subcrp_record_dict["included_variable"] = []
        return subcrp_record_dict

    def update_subcrp_record(self, subcrp_record_dict):
        subcrp_record_dict["included_variable"].append(np.sum(self.mask))
        return subcrp_record_dict


    def collapsed_gibbs_sampler(self, n_iter, true_assignments, num_saved=3, weight_first=True, burnin_mask=500,
                                mask_update='gibbs'):
        """
        Perform `n_iter` iterations Gibbs sampling on the SubCRPMM.

        A record dict is constructed over the iterations, which contains
        several fields describing the sampling process. Each field is described
        by its key and statistics are given in a list which covers the Gibbs
        sampling iterations. This dict is returned.

        Also a distribution dict is conducted when the component number equal to 'num_saved' and returned

        :param n_iter: number of sampling iterations
        :param true_assignments: true clustering assignments
        :param num_saved: save the distribution when components equal to num_saved
        :param weight_first: label switch by weight vector or by mean. By weight vector default
        :param burnin_mask: do not select variable in the first iterations
        :param mask_update: 'gibbs' for using Gibbs sampling to update mask vector
                            'metropolis' for using metropolis search to update mask vector
        :return: record dictionary & distribution dictionary
        """

        # Setup record dictionary
        record_dict = self.setup_record_dict()
        start_time = time.time()
        distribution_dict = self.setup_distribution_dict(num_saved)
        subcrp_record_dict = self.setup_subcrp_record()

        # Loop over iterations
        for i_iter in range(n_iter):

            ## save the wanted distribution
            if num_saved == self.components.K and i_iter > 1:
                distribution_dict = self.update_distribution_dict(distribution_dict, weight_first)


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
            ## End Step 1. end loop data


            ## Step 2. update dimension mask
            if i_iter > burnin_mask:
                if mask_update == 'gibbs':
                    # self.mask = np.random.binomial(1, 1, self.D)
                    self.gibbs_update_mask(i_iter)
                elif mask_update == 'metropolis':
                    self.metropolis_update_mask(i_iter)
                else:
                    assert False, "Invalid update method for mask vector."



            ## Update record
            record_dict = self.update_record_dict(record_dict, i_iter, true_assignments, start_time)
            start_time = time.time()
            subcrp_record_dict = self.update_subcrp_record(subcrp_record_dict)

        ## End loop iteration

        return record_dict, distribution_dict, subcrp_record_dict