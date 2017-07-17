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

from ..ars.ars import ARS
from fgmm_hyperprior_func import *

from ..gmm import GMM

logger = logging.getLogger(__name__)


#-----------------------------------------------------------------------------#
#                                 FGMM CLASS                                 #
#-----------------------------------------------------------------------------#

class FGMM(GMM):
    """
    Bayesian inference for a finite Gaussian mixture model (FGMM).

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
        super(FGMM, self).__init__()

        data_shape = X.shape
        if len(data_shape) < 2:
            raise ValueError('X must be at least a 2-dimensional array.')

        self.K = K
        self.old_mean = np.zeros((self.K,))
        self.old_sigma = np.zeros((self.K,))

        N, D = X.shape

        if dir_prior.name == 'symmetric-dirichlet':
            self.alpha = dir_prior.alpha
            self.kalpha = dir_prior.alpha * K

            self.hyperprior = None  ## use fixed symmetric Dirichlet prior
        elif dir_prior.name == 'hyper-dirichlet':
            self.alpha = 1. / K # default alpha for hyper dirichlet
            self.kalpha = 1.
            self.hyperprior = dir_prior
            self.a = dir_prior.a  ## hyper prior Gamma(a,b)
            self.b = dir_prior.b
            self.xi = dir_prior.xi  ## starting points for adaptive rejection sampling (ARS)
        else:
            assert False, "Invalid Dirichlet prior."

        # Initial component assignments
        if assignments == "rand":
            assignments = np.random.randint(0, K, N)

            # Make sure we have consequetive values
            for k in xrange(assignments.max()):
                while len(np.nonzero(assignments == k)[0]) == 0:
                    assignments[np.where(assignments > k)] -= 1
                if assignments.max() == k:
                    break
        elif assignments == "each-in-own":
            assignments = np.arange(N)
        else:
            # assignments is a vector
            pass

        if covariance_type == "full":
            self.components = GaussianComponents(X, kernel_prior, assignments, K_max=K)
        elif covariance_type == "diag":
            self.components = GaussianComponentsDiag(X, kernel_prior, assignments, K_max=K)
        elif covariance_type == "fixed":
            self.components = GaussianComponentsFixedVar(X, kernel_prior, assignments, K_max=K)
        else:
            assert False, "Invalid covariance type."

    def log_marg(self):
        """Return log marginal of data and component assignments: p(X, z)"""

        # Log probability of component assignment, (24.24) in Murphy, p. 842
        log_prob_z = (
            gammaln(self.kalpha)
            - gammaln(self.kalpha + np.sum(self.components.counts))
            + np.sum(
                gammaln(
                    self.components.counts
                    + float(self.kalpha)/self.components.K_max
                    )
                - gammaln(self.kalpha/self.components.K_max)
                )
            )

        # Log probability of data in each component
        log_prob_X_given_z = self.components.log_marg()

        return log_prob_z + log_prob_X_given_z


    def setup_distribution_dict(self, num_saved, n_iter):
        distribution_dict = {}
        distribution_dict["mean"] = np.zeros(shape=(num_saved, n_iter))
        distribution_dict["variance"] = np.zeros(shape=(num_saved, n_iter))
        distribution_dict["weights"] = np.zeros(shape=(num_saved, n_iter))
        return distribution_dict

    def update_distribution_dict(self, distribution_dict, weight_first, i_iter):
        ## get mean, sd and weights
        means = []
        sds = []
        for k in xrange(self.components.K):
            mu, sigma = self.components.rand_k(k)
            means.append(mu)
            sds.append(sigma)

        for i in range(self.K - self.components.K):
            means.append(self.old_mean[k])
            sds.append(self.old_sigma[k])

        ## label switch by weight first or mean first
        if weight_first:
            ## label switching index
            weights = self.gibbs_weight()
            idx = np.argsort(weights)

            sds = np.array(sds).flatten()
            means = np.array(means).flatten()

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

        distribution_dict["weights"][:, i_iter] = weights
        distribution_dict["mean"][:, i_iter] = means
        distribution_dict["variance"][:, i_iter] = sds

        ## NOTE: return the weights for the update of Hyperprior on Dirichlet by ARS
        return distribution_dict, weights

    def gibbs_weight(self):
        """
        Get weight vector for each gibbs iteration
        :return: weight vector
        """
        Nk = self.components.counts[:self.components.K].tolist()
        for i in range(self.K - self.components.K):
            Nk.append(0)
        alpha = [Nk[cid] + self.kalpha / self.K
                 for cid in range(self.K)]
        # alpha = [Nk[cid] + self.kalpha
        #          for cid in range(self.K)]

        return stats.dirichlet(alpha).rvs(size=1).flatten()


    def collapsed_gibbs_sampler(self, n_iter, true_assignments, weight_first=True):
        """
        Perform `n_iter` iterations collapsed Gibbs sampling on the FGMM.

        A record dict is constructed over the iterations, which contains
        several fields describing the sampling process. Each field is described
        by its key and statistics are given in a list which covers the Gibbs
        sampling iterations. This dict is returned.
        :param n_iter: number of iteration for Gibbs sampling
        :param true_assignments: true assignments of clustering
        :param weight_first: Label switch by weight or by mean (1D)
        :return: record_dict and distribution_dict for all records during sampling
        """

        # Setup record dictionary
        record_dict = self.setup_record_dict()
        start_time = time.time()
        distribution_dict = self.setup_distribution_dict(self.K, n_iter)


        # Loop over iterations
        for i_iter in range(n_iter):

            # Loop over data items
            for i in xrange(self.components.N):

                # Cache some old values for possible future use
                k_old = self.components.assignments[i]
                K_old = self.components.K
                stats_old = self.components.cache_component_stats(k_old)

                # Remove data vector `X[i]` from its current component
                self.components.del_item(i)

                # Compute log probability of `X[i]` belonging to each component
                # (24.26) in Murphy, p. 843
                log_prob_z = (
                    np.ones(self.components.K_max)*np.log(
                        float(self.kalpha)/self.components.K_max + self.components.counts
                        )
                    )
                # (24.23) in Murphy, p. 842
                log_prob_z[:self.components.K] += self.components.log_post_pred(i)
                # Empty (unactive) components
                log_prob_z[self.components.K:] += self.components.log_prior(i)
                prob_z = np.exp(log_prob_z - logsumexp(log_prob_z))

                # Sample the new component assignment for `X[i]`
                k = utils.draw(prob_z)

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



            # Update record
            record_dict = self.update_record_dict(record_dict, i_iter, true_assignments, start_time)
            start_time = time.time()
            distribution_dict, weights = self.update_distribution_dict(distribution_dict, weight_first, i_iter=i_iter)

            ## update for hyper prior on concentration parameter alpha by ARS
            if self.hyperprior is not None:
                weight_prod = np.prod(weights)
                K = self.components.K
                ars = ARS(f_hyperprior_log, f_hyperprior_log_prima, xi=self.xi, lb=0,
                          weight_prod=weight_prod, K=K, a=self.a, b=self.b)
                if not ars.no:
                    self.alpha = ars.draw(3)[2]
                    self.kalpha = self.alpha * self.K
                else:
                    self.kalpha = self.kalpha
                    # distribution_dict["alpha"][i_iter] = self.kalpha

        return record_dict, distribution_dict


