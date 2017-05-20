"""
The :mod:`bgmm.igmm` module implements infinite gaussian mixture modeling algorithms.
"""

# from .gmm import sample_gaussian, log_multivariate_normal_density
# from .gmm import GMM, distribute_covar_matrix_to_match_covariance_type
# from .gmm import _validate_covars
# from .dpgmm import DPGMM, VBGMM

from .crpmm import CRPMM
from .pcrpmm import PCRPMM


__all__ = ['CRPMM',
           'PCRPMM',
          ]
