"""
The :mod:`pybgmm.igmm` module implements infinite gaussian mixture modeling algorithms.
"""



from .crpmm import CRPMM
from .pcrpmm import PCRPMM
from .adapcrpmm import ADAPCRPMM
from .cscrpmm import CSCRPMM
from .subcrpmm import SubCRPMM

__all__ = ['CRPMM',
           'PCRPMM',
           'ADAPCRPMM',
           'CSCRPMM',
           'SubCRPMM',
          ]
