"""
The :mod:`pybgmm.dist` module implements some distributions.
"""

from .Dirichlet import Dirichlet
from .gDirichlet import gDirichlet

__all__ = ['Dirichlet',
           'gDirichlet',
          ]