"""
The :mod:`pybgmm.fgmm` module implements finite gaussian mixture modeling algorithms.
"""

from .fgmm import FGMM
from .pfgmm import PFGMM

__all__ = ['FGMM',
           'PFGMM',
          ]