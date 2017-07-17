import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import os
from scipy.stats import gamma as gamma_dist
import random
import scipy.integrate as integrate


class Dirichlet(object):
    def __init__(self, alpha):
        from math import gamma
        from operator import mul
        self._alpha = np.array(alpha)
        self._coef = gamma(np.sum(self._alpha)) / \
                     reduce(mul, [gamma(a) for a in self._alpha])
    def pdf(self, x):
        '''Returns pdf value for `x`.'''
        from operator import mul
        return self._coef * reduce(mul, [xx ** (aa - 1)
                                         for (xx, aa)in zip(x, self._alpha)])

    def rvs(self, size=1):
        res = np.zeros(shape=(size, self._alpha.shape[0]))
        for i in range(size):
            res[i,:] = np.random.dirichlet(self._alpha, 1)
        return res