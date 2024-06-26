```@meta
CurrentModule = FaSTLMMlight
```

# FaST-LMM full rank

## Spectral decomposition of the kernel matrix and data rotaton

We first consider the scenario where ``K`` is defined by a square semi-positive definite matrix and ``K_{22}`` has been computed by [`svd_fixed_effects`](@ref). Following [Lippert et al. (2011)](https://europepmc.org/article/med/21892150), we call this the "full rank" scenario, although neither ``K`` nor ``K_{22}`` actually has to have full rank.

The spectral decomposition of ``K_{22}`` can be written as ``K_{22}=U \Lambda U^T``, with ``U\in\mathbb{R}^{(n-d)\times (n-d)}`` unitary and ``\Lambda\in\mathbb{R}^{(n-d)\times (n-d)}`` diagonal with non-negative diagonal elements ``\lambda_i=\Lambda_{ii}\geq 0``, which we assume are ordered, ``\lambda_1\geq\lambda_2\geq \dots \geq \lambda_{n-d}\geq 0``

The columns of ``U`` are the eigenvectors of ``K_{22}``, and we denote them as ``u_i \in\mathbb{R}^{n-d}``, with ``K_{22} u_i=\lambda_i u_i``. The matrix $U$ can be used to rotate the data ``y_2``:

```math
\tilde{y} = U^T y_2 \in \mathbb{R}^{n-d}
```

with components

```math
\tilde{y}_i = \langle u_i,y_2\rangle,\quad i=1,\dots,n-d
```

## Likelihood function and variance parameter estimation

These results allow to express the terms in ``\mathcal{\ell}_R`` 

```math
\begin{aligned}
   \frac1{n-d}\log\det(K_{22}+\delta I) &= \frac1{n-d}\sum_{i=1}^{n-d} \log(\lambda_i+\delta)\\
  \hat\sigma^2 = \frac1{n-d}\langle y_2,(K_{22}+\delta I)^{-1} y_2\rangle &= \frac1{n-d}\sum_{i=1}^{n-d} \frac1{\lambda_i+\delta} \langle y_2,u_i\rangle^2 = \frac1{n-d}\sum_{i=1}^{n-d} \frac{\tilde{y}_i^2}{\lambda_i+\delta}
\end{aligned}
```

Note the crucial difference with the original FaST-LMM method. There the eigenvalue decomposition of the *full* matrix ``K`` is used to facilitate the computation of the negative log-likelihood ``\mathcal{L}`` (see [Introduction](@ref)), which still involves the solution of a linear system to compute ``\hat\beta``. By working in the restricted space orthogonal to the fixed effects and using the eigenvalue decomposition of the *reduced* matrix ``K_{22}``, we have obtained a restricted negative log-likelihood ``\ell_R`` which, given ``\lambda`` and ``\tilde y`` is trivial to evaluate and differentiate by  [automatic differentiation](https://julianlsolvers.github.io/Optim.jl/stable/user/gradientsandhessians/#Automatic-differentiation).

The values of ``\hat\sigma^2`` and ``\ell_R``  as a function of the parameter ``\delta`` and vectors ``\lambda`` and ``\tilde y`` are computed by the functions [`sigma2_reml`](@ref) and [`neg_log_like`](@ref)

```@docs
sigma2_reml
neg_log_like
```

The optimization of the function ``\ell_R`` is done by the function [`delta_reml`](@ref) using the [LBFGS algorithm](https://julianlsolvers.github.io/Optim.jl/stable/algo/lbfgs/). Because ``\delta`` must be greater than zero, we write ``\delta`` as the [`softplus`](@ref) function of an unconstrained variable ``x``, that is ``\delta=\log(1+e^x)``, and optimize with respect to ``x``

```@docs
delta_reml
softplus
```

```@docs
fastlmm_fullrank
```

## Fixed effects weights using the rotated basis

The fixed effects weights need to be computed only once, after ``\hat\delta`` has been estimated, ``\hat\beta(\hat\delta)`` is computed using the formula in [Fixed effects weights](@ref). Using the eigendecomposition of ``K_{22}`` and the rotated data ``\tilde y = U^Ty_2`` we obtain

```math
\begin{aligned}
\hat \beta(\hat\delta) &= W \Gamma^{-1} ( y_1 - K_{12} (K_{22}+\hat\delta I)^{-1} y_2 )\\
&= W \Gamma^{-1} ( y_1 - K_{12} U(\Lambda+\hat\delta I)^{-1} U^Ty_2 )\\
&= W \Gamma^{-1} ( y_1 - K_{12} U (\Lambda+\hat\delta I)^{-1} \tilde y )
\end{aligned}
```

The components of the vector ``(\Lambda+\hat\delta I) \tilde y`` are easily computed elementwise as ``\tilde y_i/(\lambda_i+\hat\delta)``.

This calculation is implemented in the function [`beta_mle`](@ref):

```@docs
beta_mle
```