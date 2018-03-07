# PyBGMM: Bayesian inference for Gaussian mixture model

More coming...

## Overview
Approximation inference (Bayesian inference) for finite Gaussian mixture model (FGMM) and infinte Gaussian mixture 
model (IGMM) includes variational inference and Monte Carlo methods. Here we only use Monte Carlo methods. In 
particular, we use collapsed Gibbs sampling to do the inference.

## Code Structure

```
|-- GMM # base class for Gaussian mixture model
    |---- FGMM  # base class for finite Gaussian mixture model
        |------ PFGMM
        |------ CSFGMM
        |------ LSFGMM

    |---- IGMM  # base class for infinite Gaussian mixture model
        |------ CRPMM
        |------ PCRPMM    ## powered Chinese restaurant process (pCRP) mixture model
        |------ CSIGMM
        |------ LSIGMM
        |------ SubCRPMM  ## Sub-clustering with CRP mixture model for high-dimensional data
```

## Documentation

What do we include:

- Finite Gaussian mixture model

- Hyperprior on Dirichlet distribution (for finite Gaussian mixture model)

- Chinese restaurant process mixture model (CRPMM)

- Powered Chinese restaurant process (pCRP) mixture model

- Adaptive powered Chinese restaurant process (Ada-pCRP) mixture model

- Constrained sampling for Chinese restaurant process mixture model

- Bayesian variable selection in Chinese restaurant process mixture (Sub-CRP)

What we will include:

- Hyperprior on Dirichlet process prior (for infinite Gaussian mixture model)

## Examples


| Code | Description |
|:-------:| ----------- |
| [CRPMM 1d](/examples/crpmm_1d_demo.py) | Chinese restaurant process mixture model for 1d data |
| [CRPMM 2d](/examples/crpmm_1d_demo.py) | Chinese restaurant process mixture model for 2d data |
| [pCRPMM 1d](/examples/pcrpmm_1d_demo.py) | powered Chinese restaurant process mixture model for 1d data |
| [pCRPMM 2d](/examples/pcrpmm_2d_demo.py) | powered Chinese restaurant process mixture model for 2d data |
| [SubCRP](/examples/subcrp_demo.py) | several test on SubCRP mixture model (Bayesian variable selection for high-dimensional data in CRP) |
| [CSIGMM](/examples/csigmm_1d_demo.py) | demo for constrained sampling for CRPMM |
| [CRP draw](/examples/crp_draw.py) | A basic demo for CRP prior draw |

## Dependencies
1. [Adaptive Rejection Sampling (ARS)](https://github.com/junlulocky/ARS-MCMC) - Python implementation of ARS.
1. [Clustering accuracy](https://github.com/junlulocky/infopy) - infopy: Python implementation of information theory 
computation.
1. See requirements.txt

## Lincense
MIT

## Citation
The repo is based on the following research articles:

- Lu, Jun, Meng Li, and David Dunson. "Reducing over-clustering via the powered Chinese restaurant process." arXiv preprint arXiv:1802.05392 (2018).
- Lu, Jun. "Robust model-based clustering for big and complex data", 2017.
- Lu, Jun. "Hyperprior on symmetric Dirichlet distribution." arXiv preprint arXiv:1708.08177 (2017).

## References
1. Lu, Jun. "Hyperprior on symmetric Dirichlet distribution." arXiv preprint arXiv:1708.08177 (2017).
1. H. Kamper, A. Jansen, S. King, and S. Goldwater, "Unsupervised lexical clustering of speech segments using 
fixed-dimensional acoustic embeddings", in Proceedings of the IEEE Spoken Language Technology Workshop (SLT), 2014.
1. Murphy, Kevin P. "Conjugate Bayesian analysis of the Gaussian distribution." def 1.2σ2 (2007): 16.
1. Murphy, Kevin P. Machine learning: a probabilistic perspective. MIT press, 2012.
1. Pedregosa, Fabian, et al. "Scikit-learn: Machine learning in Python." Journal of Machine Learning Research 12.Oct
 (2011): 2825-2830.
1. Rasmussen, Carl Edward. "The infinite Gaussian mixture model." Advances in neural information processing systems. 2000.
1. Tadesse, Mahlet G., Naijun Sha, and Marina Vannucci. "Bayesian variable selection in clustering high-dimensional data." Journal of the American Statistical Association 100.470 (2005): 602-617.
