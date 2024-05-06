```@meta
CurrentModule = FaSTLMMlight
```

# FaST-LMM light Documentation


## Introduction

We consider the setup of [Lippert et al.][1], where:

- ``y\in\mathbb{R}^n`` is a phenotype vector with values for ``n`` individuals,
- ``X\in\mathbb{R}^{n\times d}`` is a matrix with data of ``d`` covariates (fixed effects) in the same ``n`` individuals,
- ``K\in\mathbb{R}^{n\times n}`` is a positive semi-definite sample similarity matrix, scaled such that ``\mathrm{tr}(K)=n``,
- ``\beta\in\mathbb{R}^d`` is the (unknown) vector of fixed effect weights,
- ``\sigma^2`` is the (unknown) variance explained by ``K``,
- ``\sigma_e^2`` is the (unknown) residual error variance,

[1]: https://europepmc.org/article/med/21892150

## Table of contents

```@contents
```