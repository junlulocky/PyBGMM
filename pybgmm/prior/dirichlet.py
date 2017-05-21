"""
Prior information for a Dirichlet distribution
Author: Jun Lu
Contact: jun.lu.locky@gmail.com
Date: 2017
"""

class Dirichlet(object):
    """A Symmetric Dichlet distribution prior."""
    def __init__(self, alpha):
        """

        :param alpha: Symmetric Dirichlet dist, i.e. Dirichlet(alpha, ..., alpha)
        """
        self.name = 'symmetric-dirichlet'
        assert alpha>0, "concentration parameter must be larger than 0"
        self.alpha = alpha

