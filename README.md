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
        |------ PCRPMM
        |------ CSIGMM
        |------ LSIGMM
```

## Documentation

What do we include:

- Finite Gaussian mixture model

- Hyperprior on Dirichlet distribution (for finite Gaussian mixture model)

- Chinese restaurant process (CRP) mixture model

- Powered Chinese restaurant process (pCRP) mixture model

- Adaptive powered Chinese restaurant process (Ada-pCRP) mixture model

- Constrained sampling for Chinese restaurant process mixture model

What we will include:

- Hyperprior on Dirichlet process prior (for infinite Gaussian mixture model)

## Dependencies
1. [Adaptive Rejection Sampling (ARS)](https://github.com/junlulocky/ARS-MCMC) - Python implementation of ARS.
1. [Clustering accuracy](https://github.com/junlulocky/infopy) - infopy: Python implementation of information theory 
computation.
1. See requirements.txt

## Lincense
MIT

## References
[1]. H. Kamper, A. Jansen, S. King, and S. Goldwater, "Unsupervised lexical clustering of speech segments using 
fixed-dimensional acoustic embeddings", in Proceedings of the IEEE Spoken Language Technology Workshop (SLT), 2014.

[2]. Murphy, Kevin P. "Conjugate Bayesian analysis of the Gaussian distribution." def 1.2σ2 (2007): 16.

[3]. Murphy, Kevin P. Machine learning: a probabilistic perspective. MIT press, 2012.

[4]. Pedregosa, Fabian, et al. "Scikit-learn: Machine learning in Python." Journal of Machine Learning Research 12.Oct
 (2011): 2825-2830.
 
[5]. Rasmussen, Carl Edward. "The infinite Gaussian mixture model." Advances in neural information processing systems. 2000.
