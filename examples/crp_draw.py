"""
A basic demo for crp prior draw.

Author: Jun Lu
Contact: jun.lu.locky@gmail.com
Date: 2017
"""

import sys
sys.path.append("..")
from pybgmm.prior.crp import CRP
from matplotlib import pyplot as plt
import numpy as np


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