# FaST-LMM light

Lightweight implementation of the [FaST-LMM algorithm](https://europepmc.org/article/med/21892150) for estimating the parameters $\beta$, $\sigma^2$ and $\delta$ in a linear [mixed model](https://en.wikipedia.org/wiki/Mixed_model) or [Gaussian process](https://en.wikipedia.org/wiki/Gaussian_process) of the form

$$y \\sim N\\bigl(X\\beta, \\sigma^2(K + \\delta I) \\bigr)$$

with $y\in\mathbb{R}^n$ an observed response vector, $X\in\mathbb{R}^{n\times d}$ an observed matrix of $d$ covariates, $K\in\mathbb{R}^{n\times n}$ a kernel matrix and $I$ the identity matrix. 

*"Lightweight"* refers to two aspects of this implementation:

1. The package is agnostic to any application domain. Whereas the [original software](https://fastlmm.github.io/) and an [existing julia implementation](https://github.com/sens/FaSTLMM.jl) contain a lot of code that is specific to working with genetic data, `FaSTLMMlight` contains only functions that work with the variables defined in the equation above, and nothing else. Thus the package provides generic functions that other packages can call to solve the parameter estimation problem (for an example, see [SpatialOmicsGPs](https://github.com/tmichoel/SpatialOmicsGPs.jl)).

2. The package introduces a new, lightweight formulation of the one-parameter cost function (after eliminating $\beta$ and $\sigma^2$). Whereas the original algorithm involves the solution of a linear system in each evaluation of the cost function (to express $\beta$ as a function of $\delta$), `FaSTLMMlight` uses [restricted maximum likelihood](https://en.wikipedia.org/wiki/Restricted_maximum_likelihood) properties to avoid this step, resulting in a cost function amenable to [automatic differentiation](https://julianlsolvers.github.io/Optim.jl/stable/user/gradientsandhessians/#Automatic-differentiation). This in turns avoids the need for any grid-based optimization.



[![Build Status](https://github.com/tmichoel/FaSTLMMlight.jl/actions/workflows/CI.yml/badge.svg?branch=master)](https://github.com/tmichoel/FaSTLMMlight.jl/actions/workflows/CI.yml?query=branch%3Amaster)
