import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import os
from scipy.stats import gamma as gamma_dist
import random
import scipy.integrate as integrate


class gDirichlet(object):
    def __init__(self, a, b, K):
        from operator import mul
        self._a = a
        self._b = b
        self._K = K
        theta = 1. / b
        self.gamma_dist = gamma_dist(a, 0, theta)
        # self._alpha = np.array(alpha)
        # self._coef = gamma(np.sum(self._alpha)) / \
        #              reduce(mul, [gamma(a) for a in self._alpha])

    def gdir_pdf_integ(self, alpha1, alpha2, alpha3):
        from math import gamma
        from operator import mul
        alpha = np.array([alpha1, alpha2, alpha3])
        coef1 = gamma(np.sum(alpha)) / reduce(mul, [gamma(a) for a in alpha])
        coef2 = reduce(mul, [xx ** (aa - 1)
                             for (xx, aa) in zip(self.x, alpha)])

        coef3 = reduce(mul, [self.gamma_dist.pdf(local) for local in alpha])
        return coef1 * coef2 * coef3

    def pdf(self, x):
        '''Returns pdf value for `x`.'''
        self.x = x
        integrate.nquad(self.gdir_pdf_integ, [[0, np.inf], [0, np.inf], [0, np.inf]])

        from operator import mul
        return self._coef * reduce(mul, [xx ** (aa - 1)
                                         for (xx, aa) in zip(x, self._alpha)])

    def rvs(self, size=1):
        res = np.zeros(shape=(size, self._K))
        for i in range(size):
            alpha = np.array([random.gammavariate(self._a, 1. / self._b) for local in range(self._K)])
            try:
                res[i,:] = np.random.dirichlet(alpha, 1)
            except:
                res[i, :] = res[i-1,:]
            # res[i, :] = np.array([random.gammavariate(local, 1) for local in alpha])

        return res