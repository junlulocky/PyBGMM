import numpy as np
from scipy import special

######################################
# hyper prior on Dirichlet distribution in FGMM
# from a Gamma(a, b)
######################################
def f_hyperprior_log(x, weight_prod, K, a=1, b=1):
    """
    Log distribution
    """
    return (a-1) * np.log(x) - b*x + (x-1)* np.log(weight_prod) + np.log(special.gamma(K*x)) - K*np.log(
        special.gamma(x))



def f_hyperprior_log_prima(x, weight_prod, K, a=1, b=1):
    """
    Derivative of Log distribution
    """
    res = (a-1) *1. / x - b + np.log(weight_prod) + K * np.log(K) - K * special.digamma(x)
    for i in range(K):
        res += special.digamma(x + i*1./K)
    return res

