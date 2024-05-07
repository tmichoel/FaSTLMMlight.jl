```@meta
CurrentModule = FaSTLMMlight
```

# FaST-LMM light Documentation


## Introduction

We consider the setup of [Lippert et al.](https://europepmc.org/article/med/21892150), where:

- ``y\in\mathbb{R}^n`` is a phenotype vector with values for ``n`` individuals,
- ``X\in\mathbb{R}^{n\times d}`` is a matrix with data of ``d`` covariates (fixed effects) in the same ``n`` individuals,
- ``K\in\mathbb{R}^{n\times n}`` is a positive semi-definite sample similarity matrix, scaled such that ``\mathrm{tr}(K)=n``,
- ``\beta\in\mathbb{R}^d`` is the (unknown) vector of fixed effect weights,
- ``\sigma^2`` is the (unknown) variance explained by ``K``,
- ``\sigma_e^2`` is the (unknown) residual error variance,
- ``\delta = \frac{\sigma_e^2}{\sigma^2}`` is the variance ratio,

and ``y`` is distributed as

```math
y \sim N\bigl( X\beta, \sigma^2 \K + \sigma_e^2 I\bigr) = N\bigl( X\beta, \sigma^2 (\K + \delta I)\bigr)
```

where ``N`` denotes a multivariate normal distribution.

The unknown parameters ``(\beta,\sigma^2,\delta)`` are estimated by maximum-likelihood (or restricted maximum-likelihood, see below), that is, by minimizing the negative log-likelihood function

```math
\mathcal{L} = \log\det\bigl[ \sigma^2 (K + \delta I) \bigr] + \frac1{\sigma^2} \bigl\langle y-X\beta, (K + \delta I)^{-1} (y-X\beta)\bigr\rangle,
```

where ``\langle u,v\rangle=u^Tv=\sum_{i=1}^n u_n v_n`` is the usual inner product on ``\mathbb{R}^n``; note that for any matrix ``A\in\mathbb{R}^{n\times n}`` and vectors ``u,v\in\mathbb{R]^n``, ``\langle u,Av\rangle=\mathrm{tr}(vu^TA)``. 

## Table of contents

```@contents
```