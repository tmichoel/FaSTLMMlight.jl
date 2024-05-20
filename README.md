# FaST-LMM light

Lightweight implementation of the [FaST-LMM algorithm](https://europepmc.org/article/med/21892150) for estimating the parameters $\beta$, $\sigma^2$ and $\delta$ in a linear [mixed model](https://en.wikipedia.org/wiki/Mixed_model) or [Gaussian process](https://en.wikipedia.org/wiki/Gaussian_process) of the form

$$y \\sim N\\bigl(X\\beta, \\sigma^2(K + \\delta I) \\bigr)$$

with $y\in\mathbb{R}^n$ an observed response vector, $X\in\mathbb{R}^{n\times d}$ an observed matrix of $d$ covariates, $K\in\mathbb{R}^{n\times n}$ a kernel matrix and $I$ the identity matrix. 

*"Lightweight"* refers to two aspects of this implementation:

1. The package is agnostic to any application domain. Whereas the [original software](https://fastlmm.github.io/) and an [existing julia implementation](https://github.com/sens/FaSTLMM.jl) contain a lot of code that is specific to working with genetic data, `FaSTLMMlight` contains only functions that work with the variables defined in the equation above, and nothing else. In other words, the package provides generic functions that other packages can call to solve the parameter estimation problem in specific application domains (for an example, see [SpatialOmicsGPs](https://github.com/tmichoel/SpatialOmicsGPs.jl)).

2. The package introduces a new, lightweight formulation of the one-parameter cost function (after eliminating $\beta$ and $\sigma^2$). Whereas the original algorithm involves the solution of a linear system in each evaluation of the cost function (to express $\beta$ as a function of $\delta$), `FaSTLMMlight` uses [restricted maximum likelihood](https://en.wikipedia.org/wiki/Restricted_maximum_likelihood) properties to avoid this step, resulting in a cost function amenable to [automatic differentiation](https://julianlsolvers.github.io/Optim.jl/stable/user/gradientsandhessians/#Automatic-differentiation). This in turns avoids the need for any grid-based optimization. See the documentation pages ["SVD of the fixed effects"](https://lab.michoel.info/FaSTLMMlight.jl/dev/svd-fixed-effects/) and ["FaST-LMM full rank"](https://lab.michoel.info/FaSTLMMlight.jl/dev/fastlmm-fullrank/) for details.

## Roadmap

 `FaSTLMMlight` will implement the FaST-LMM algorithms for full-rank kernel matrices and low-rank kernel matrix factorizations. I am also investigating whether and when the Fourier-space method of [Greengard et al.](https://arxiv.org/abs/2210.10210) can lead to further speed-up.

 Currently completed:

- [x] LMMs/GPs with full-rank kernel matrix
- [ ] LMMs/GPs with low-rank kernel matrix factorization
- [ ] Fourier-space method for translationally invariant kernels

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://tmichoel.github.io/FaSTLMMlight.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://tmichoel.github.io/FaSTLMMlight.jl/dev/)
[![Build Status](https://github.com/tmichoel/FaSTLMMlight.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/tmichoel/FaSTLMMlight.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/tmichoel/FaSTLMMlight.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/tmichoel/FaSTLMMlight.jl)