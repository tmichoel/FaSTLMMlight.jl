```@meta
CurrentModule = FaSTLMMlight
```

# FaST-LMM full rank


The spectral decomposition of ``K_{22}`` can be written as ``K_{22}=U \Lambda U^T``, with ``U\in\mathbb{R}^{(n-d)\times (n-d)}`` unitary and ``\Lambda\in\mathbb{R}^{(n-d)\times (n-d)}`` diagonal with non-negative diagonal elements ``\lambda_i=\Lambda_{ii}\geq 0``. In general, ``K_{22}`` need not be full rank, we assume that ``\mathrm{rank}(K_{22})=r\leq n-d``, and that the eigenvalues are ordered, ``\lambda_1\geq\lambda_2\geq \dots \geq \lambda_r > \lambda_{r+1} = \dots = \lambda_{n-d} = 0``. We can hence also write ``K_{22} = U_1\Lambda_1U_1^T``, where ``U_1\in\mathbb{R}^{(n-d)\times r}`` with ``U_1^TU_1=I``, ``\Lambda_1\in\mathbb{R}^{r\times r}`` diagonal with diagonal elements ``\lambda_1,\dots,\lambda_r``, and ``U = (U_1, U_2)`` with ``U_2^TU_2=I``, ``U_1^TU_2=0`` and ``U_1U_1^T+U_2U_2^T=I``. The columns of ``U`` are the eigenvectors of ``K_{22}``, and we denote them as ``u_i \in\mathbb{R}^{n-d}``, with ``K_{22} u_i=\lambda_i u_i``. These results allow to express the terms in ``\mathcal{L}_R`` as

```math
\begin{aligned}
   \log\det(K_{22}+\delta I) &= \sum_{i=1}^r \log(\lambda_i+\delta) + (n-d-r)\log(\delta)\\
  \langle y_2,(K_{22}+\delta I)^{-1} y_2\rangle &= \sum_{i=1}^r \frac1{\lambda_i+\delta} \langle y_2,u_i\rangle^2 + \frac1{\delta} \langle y_2,U_2U_2^Ty_2\rangle\\
  &= \sum_{i=1}^r \frac1{\lambda_i+\delta} \langle y_2,u_i\rangle^2 + \frac1{\delta} \|U_2^Ty_2\|^2.
\end{aligned}
```


```@docs
fastlmm_fullrank
delta_mle_fullrank
minus_log_like_fullrank
sigma2_mle_fullrank
beta_mle_fullrank_lazy
create_covariate_matrix
project_orth_covar
softplus
```