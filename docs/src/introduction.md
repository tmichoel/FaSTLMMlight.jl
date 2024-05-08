```@meta
CurrentModule = FaSTLMMlight
```

# Introduction

We consider the setup of [Lippert et al. (2011)](https://europepmc.org/article/med/21892150), where:

- ``y\in\mathbb{R}^n`` is a response vector with values for ``n`` samples,
- ``X\in\mathbb{R}^{n\times d}`` is a matrix with data of ``d`` covariates (fixed effects) in the same ``n`` samples,
- ``K\in\mathbb{R}^{n\times n}`` is a positive semi-definite sample similarity matrix, scaled such that ``\mathrm{tr}(K)=n``,
- ``\beta\in\mathbb{R}^d`` is the (unknown) vector of fixed effect weights,
- ``\sigma^2`` is the (unknown) variance explained by ``K``,
- ``\sigma_e^2`` is the (unknown) residual error variance,
- ``\delta = \frac{\sigma_e^2}{\sigma^2}`` is the variance ratio,

and ``y`` is distributed as

```math
y \sim N\bigl( X\beta, \sigma^2 K + \sigma_e^2 I\bigr) = N\bigl( X\beta, \sigma^2 (K + \delta I)\bigr)
```

where ``N`` denotes a multivariate normal distribution.

The unknown parameters ``(\beta,\sigma^2,\delta)`` are estimated by maximum-likelihood (or restricted maximum-likelihood, see below), that is, by minimizing the negative log-likelihood function

```math
\mathcal{L} = \log\det\bigl[ \sigma^2 (K + \delta I) \bigr] + \frac1{\sigma^2} \bigl\langle y-X\beta, (K + \delta I)^{-1} (y-X\beta)\bigr\rangle,
```

where ``\langle u,v\rangle=u^Tv=\sum_{i=1}^n u_n v_n`` is the usual inner product on ``\mathbb{R}^n``; note that for any matrix ``A\in\mathbb{R}^{n\times n}`` and vectors ``u,v\in\mathbb{R]^n``, ``\langle u,Av\rangle=\mathrm{tr}(vu^TA)``. 

Analytic expressions for the maximum-likelihood estimates ``\hat{\beta}`` and ``\hat{\sigma}^2`` as a function of ``\delta`` are easy to obtain:

```math
\begin{aligned}
\hat\beta &= \bigl[ X^T(K+\delta I)^{-1}X\bigr]^{-1}
  X^T(K+\delta I)^{-1} y \\
  \hat{\sigma}^2 &= \frac1n \bigl\langle y-X\hat\beta, (K+\delta I)^{-1} (y-X\hat\beta)\bigr\rangle, 
\end{aligned}
```

Plugging these expressions into the negative log-likelihood results in a (non-convex) function of the parameter ``\delta``, which upto an additive constant is given by: 

```math
\mathcal{L}(\delta) = \log\det (K+\delta I) + n \log \hat{\sigma}^2
```

In [Lippert et al. (2011)](https://europepmc.org/article/med/21892150), the eigenvalue decomposition of ``K`` is used to increase the efficiency of evaluating ``\mathcal{L}(\delta)`` for a range of ``\delta`` values, and thereby speed up the process of finding the maximum-likelihood estimate ``\hat\delta``. However, their method still expresses the evaluation of ``\hat\beta(\delta)`` as the solution of a linear system (if the number of covariates ``d>1``). This implies that the gradient of ``\mathcal{L}(\delta)`` cannot be computed using [automatic differentiation](https://julianlsolvers.github.io/Optim.jl/stable/user/gradientsandhessians/#Automatic-differentiation), which means that only [gradient-free optimization algorithms](https://julianlsolvers.github.io/Optim.jl/stable/algo/nelder_mead/) can be used ([Lippert et al. (2011)](https://europepmc.org/article/med/21892150) used a basic grid-based search).

Instead, [FaST-LMM light](https://github.com/tmichoel/FaSTLMMlight.jl) first uses the singular value decomposition of the fixed effects matrix ``X`` to express the negative log-likelihood on the space orthogonal to the columns of ``X`` (in other words, uses [restricted maximum likelihood](https://en.wikipedia.org/wiki/Restricted_maximum_likelihood)). Then the spectral decomposition of ``K`` on the restricted space is used in the same manner as in the original FaST-LMM method, which results in a restricted negative log-likelihood function ``\mathcal{L}_R(\delta)`` whose gradient *can* be evaluated using [automatic differentiation](https://julianlsolvers.github.io/Optim.jl/stable/user/gradientsandhessians/#Automatic-differentiation).