# PyBGMM: Bayesian inference for Gaussian mixture model


## Overview
Approximation inference (Bayesian inference) for finite Gaussian mixture model (FGMM) and infinte Gaussian mixture 
model (IGMM) includes variational inference and Monte Carlo methods. Here we only use Monte Carlo methods. In 
particular, we use collapsed Gibbs sampling to do the inference.

## Code Structure

```
|-- GMM # base class for Gaussian mixture model
    |---- IGMM  # base class for infinite Gaussian mixture model
        |------ CRPMM     ## traditional Chinese restaurant process (CRP) mixture model
        |------ PCRPMM    ## powered Chinese restaurant process (pCRP) mixture model
```

## Documentation

What do we include:
- Chinese restaurant process mixture model (CRPMM)

- Powered Chinese restaurant process (pCRP) mixture model


## Examples


| Code | Description |
|:-------:| ----------- |
| [CRPMM 1d](/examples/crpmm_1d_demo.py) | Chinese restaurant process mixture model for 1d data |
| [CRPMM 2d](/examples/crpmm_1d_demo.py) | Chinese restaurant process mixture model for 2d data |
| [pCRPMM 1d](/examples/pcrpmm_1d_demo.py) | powered Chinese restaurant process mixture model for 1d data |
| [pCRPMM 2d](/examples/pcrpmm_2d_demo.py) | powered Chinese restaurant process mixture model for 2d data |


## Dependencies
1. See requirements.txt

## Lincense
MIT

## Citation
The repo is based on the following research articles:

- Lu, Jun, Meng Li, and David Dunson. "Reducing over-clustering via the powered Chinese restaurant process." arXiv preprint arXiv:1802.05392 (2018).

## References
1. H. Kamper, A. Jansen, S. King, and S. Goldwater, "Unsupervised lexical clustering of speech segments using 
fixed-dimensional acoustic embeddings", in Proceedings of the IEEE Spoken Language Technology Workshop (SLT), 2014.
1. Murphy, Kevin P. "Conjugate Bayesian analysis of the Gaussian distribution." def 1.2σ2 (2007): 16.
1. Murphy, Kevin P. Machine learning: a probabilistic perspective. MIT press, 2012.
1. Pedregosa, Fabian, et al. "Scikit-learn: Machine learning in Python." Journal of Machine Learning Research 12.Oct
 (2011): 2825-2830.
1. Rasmussen, Carl Edward. "The infinite Gaussian mixture model." Advances in neural information processing systems. 2000.
1. Tadesse, Mahlet G., Naijun Sha, and Marina Vannucci. "Bayesian variable selection in clustering high-dimensional data." Journal of the American Statistical Association 100.470 (2005): 602-617.
