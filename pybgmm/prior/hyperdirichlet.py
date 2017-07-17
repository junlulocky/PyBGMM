"""
Prior information for a Hyper Dirichlet distribution
Author: Jun Lu
Contact: jun.lu.locky@gmail.com
Date: 2017
"""

class HyperDirichlet(object):
    """A Hyper Dichlet distribution prior."""
    def __init__(self, a, b, xi=[0.1, 1, 10]):
        """

        :param a, b: Hyperprior (Gamma(a,b)) on Dirichlet dist,
                 i.e. alpha \sim Gamma(a,b)
                      Dirichlet(alpha, ..., alpha)
        :param xi: an array to indicate the starting points of ARS
        """
        self.name = 'hyper-dirichlet'
        assert a>=1, "a must larger or equal to 1"
        self.a = a
        self.b = b
        self.xi = xi

