"""
Prior information for a beta distribution
Author: Jun Lu
Contact: jun.lu.locky@gmail.com
Date: 2017
"""

class BetaBern(object):
    def __init__(self, a, b):
        """

        :param a,b: Beta(a,b) prior on Bernoulli distribution
        """
        self.name = 'Beta'
        assert a>=0, "a must larger or equal to 0"
        self.a = a
        self.b = b

