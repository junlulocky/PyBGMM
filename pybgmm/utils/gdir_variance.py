

from scipy import special
import math
import numpy as np
import matplotlib.pyplot as plt
import os
# import sympy.functions.special.gamma_functions.uppergamma
from sympy.functions.special.gamma_functions import uppergamma

def upper_incomplete_gamma_function(s,x):
    return special.gammaincc(s,x) * special.gamma(s)


def compute_symmetric_dir_variance(K, alpha):
    """
    compute the marginal distribution of \pi_1 for symmetric Dirichlet distribution
    :param K: dimension
    :param alpha: concentration paramegter for Dirichlet(alpha, ..., alpha)
    :return: variance of marginal distribution of \pi_1
    """
    common = 1. * (K - 1) / (pow(K, 3) * alpha + pow(K, 2))

    return common

def compute_gdir_variance(K, a, b):
    """
    compute the marginal distribution of \pi_1 for gDirichlet distribution
    :param K: dimension
    :param a, b: Gamma(a,b) prior on concentration parameter in Dirichlet distribution
    :return:
    """
    common = 1. * (K-1)/(pow(K,3)* a + pow(K,2))

    uncommon = K * a * pow(b, K*a) * math.exp(b) * float(uppergamma(1-K*a, b))

    return common * (1 + uncommon)

######  example ########
if __name__ == '__main__':
    print compute_symmetric_dir_variance(4, 0.15)
    print compute_symmetric_dir_variance(4, 0.3)
    print compute_gdir_variance(4, 0.3, 0.3)
    print compute_gdir_variance(4, 0.3, 3)


    def plot_variance_cuve():
        ################# plot for variance ##########################
        K = 5
        alpha = a = 0.3
        all_num = 1000
        b = np.linspace(0, 20, num=all_num)

        # print upper_incomplete_gamma_function(0.1, 1)
        # print uppergamma(0.1, 1)
        # a = uppergamma(0.1, 1)
        # print float(a)
        # print compute_gdir_variance(K, a, 1)

        symmetric_dir_var = [compute_symmetric_dir_variance(K, alpha)] * all_num
        gdir_var = [compute_gdir_variance(K, a, local_b) for local_b in b]
        # print gdir_var

        save_path = os.path.dirname(__file__) + '/res_gdir/res_variance'


        plt.figure(1)

        plt.rc('xtick', labelsize=20)
        plt.rc('ytick', labelsize=20)
        plt.tick_params(axis='both', which='major', labelsize=20)
        plt.plot(b, gdir_var)
        plt.plot(b, symmetric_dir_var)
        plt.plot((a, a), (0, 1./K), 'k-')

        plt.savefig(save_path + '/gdir_K{}_a{}.png'.format(K, a))
        plt.savefig(save_path + '/gdir_K{}_a{}.pdf'.format(K, a))



        plt.show()
