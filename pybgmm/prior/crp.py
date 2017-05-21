"""
Chinese restaurant process prior
Author: Jun Lu
Contact: jun.lu.locky@gmail.com
Date: 2017
"""

import numpy as np
import scipy as sp
import random
from matplotlib import pyplot as plt

class CRP:
    """Chinese Restaurant Process
    ref: chrome-extension://klbibkeccnjlkjkiokjodocebajanakg/suspended.html#uri=https://www.cs.princeton.edu/courses/archive/fall07/cos597C/scribe/20070921.pdf
    """

    def __init__(self, n, alpha=3, G_0='normal', name=None):
        self._n = n
        self._alpha = alpha
        self._results = []
        self._results.append(np.random.normal(1))

        self._tables = [1]  # First customer always sits at the first table

    def __call__(self):
        self.process_simulation()
        self.table_simulation()

    def process_simulation(self):
        """
        draw samples from base function
        :return:
        """
        for i in range(self._n):
            probability = self._alpha / float(self._alpha + i - 1)
            tmp = np.random.uniform(size=(1,))
            if tmp < probability:
                self._results.append(np.random.normal(1))
            else:
                self._results.append(np.random.choice(self._results[:i-1], 1)[0])

    def table_simulation(self):
        """
        draw table & cluster numbers
        :return:
        """

        # n=1 is the first customer
        for n in range(2, self._n):

            # Gen random number 0~1
            rand = random.random()

            p_total = 0
            existing_table = False

            for k, n_k in enumerate(self._tables):

                prob = n_k / (n + self._alpha - 1)

                p_total += prob
                if rand < p_total:
                    self._tables[k] += 1
                    existing_table = True
                    break

            # New table!!
            if not existing_table:
                self._tables.append(1)

        return self._tables

        # for k, n_k in enumerate(self._tables):
        #     print k, "X" * (n_k / 100), n_k
        # print "----"

    @property
    def process_results(self):
        return self._results

    @property
    def table_results(self):
        return self._tables

    @property
    def name(self):
        return self._name


if __name__ == '__main__':


    crp = CRP(n=5000, alpha=1.0)
    crp.table_simulation()

    crp_tables = crp.table_results
    for k, n_k in enumerate(crp_tables):
        print k, "X" * (n_k / 100), n_k
    print len(crp_tables)
    print "----"

    # plt.subplots_adjust(hspace=0.4)
    t = np.arange(1, 10e5, 500).astype(int)


    res = [len(CRP(n=local, alpha=1.0).table_simulation()) for local in t]
    print res
    plt.figure()
    plt.semilogx(t, res)
    plt.title('semilogx')
    plt.grid(True)



    plt.show()