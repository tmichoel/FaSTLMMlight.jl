```@meta
CurrentModule = FaSTLMMlight
```

# SVD of the fixed effects

We assume that the number of fixed effects is less than the number of  samples, ``d < n``, and that the matrix ``X`` of fixed effects has full rank, ``\mathrm{rank}(X)=d``. We can then write the "thin" singular value decomposition of ``X`` as ``X = V_1 \Gamma W^T``, where ``V_1\in\mathbb{R}^{n\times d}`` with ``V_1^TV_1=I``, ``\Gamma \in\mathbb{R}^{d\times d}`` diagonal with ``\gamma_i=\Gamma_{ii}>0`` for all ``i``, and ``W\in\mathbb{R}^{d\times d}`` with ``W^TW=WW^T=I``. Furthermore there exists ``V_2\in\mathbb{R}^{n\times (n-d)}`` with ``V_2^TV_2=I``, ``V_1^TV_2=0`` and ``V_1V_1^T+V_2V_2^T=I``, i.e. the matrix ``V=(V_1, V_2)`` is unitary, and ``V_1V_1^T`` and ``V_2V_2^T`` are the orthogonal projection matrices on the range and (left) null space of ``X``, respectively.

Using the singular value decomposition of ``X``, we can write any matrix ``A\in\mathbb{R}^{n\times n}`` and vector ``y\in\mathbb{R}^n`` in block matrix/vector notation as

```math
\begin{aligned}
  A &=
       \begin{pmatrix}
         A_{11} & A_{12}\\
         A_{21} & A_{22}
       \end{pmatrix},\;\text{where}\quad
        A_{ij} = V_i^T A V_j\\
  y &=
      \begin{pmatrix}
        y_1\\ y_2 
      \end{pmatrix} ,\;\text{where}\quad
  y_i = V_i^T y.
\end{aligned}
```

Using standard results for the [inverse](https://en.wikipedia.org/wiki/Block_matrix#Inversion) and [determinant](https://en.wikipedia.org/wiki/Block_matrix#Determinant) of a [block matrix](https://en.wikipedia.org/wiki/Block_matrix), we have

```math
  \log\det(A) = \log\det(A_{11}) + \log\det(A_{22}-A_{21}A_{11}^{-1}A_{12}) = \log \det(A_{11}) - \log\det([A^{-1}]_{22})
```

Furthermore, we have

```math
  X (X^T A X)^{-1} X^T = V_1\Gamma W^T (W\Gamma V_1^T A V_1\Gamma W^T)^{-1} W \Gamma V_1^T  = V_1 (V_1^T A V_1)^{-1} V_1^T = V_1 (A_{11})^{-1} V_1^T.
```

Using this for the specific case where ``A=(K+\delta I)^{-1}``, together with the identity ``y=(V_1V_1^T+V_2V_2^T)y = V_1y_1 + V_2y_2``, in the equation 

```math
  \bigl\langle y-X\hat\beta, A (y-X\hat\beta)\bigr\rangle= \langle y,A y\rangle - \langle y, A X (X^TAX)^{-1} X^T A y\rangle
  = \bigl\langle y, \bigl(A - A X (X^TAX)^{-1} X^T A\bigr) y \bigr\rangle,
```

gives

```math
\begin{aligned}
  &\bigl\langle y-X\hat\beta, A (y-X\hat\beta)\bigr\rangle\\
  \qquad&=
                    \begin{pmatrix}
                      y_1^T &  y_2^T
                    \end{pmatrix}
                    \begin{pmatrix}
                      A_{11} & A_{12}\\
                      A_{21} & A_{22}
                    \end{pmatrix}
                    \begin{pmatrix}
                      y_1\\ y_2
                    \end{pmatrix}
  -\begin{pmatrix}
    y_1^T &  y_2^T
  \end{pmatrix}
   \begin{pmatrix}
     A_{11} (A_{11})^{-1} A_{11} & A_{11} (A_{11})^{-1} A_{12}\\
     A_{21}A_{11} (A_{11})^{-1} & A_{21} (A_{11})^{-1}A_{12}
   \end{pmatrix}
   \begin{pmatrix}
     y_1\\ y_2
   \end{pmatrix} \\
  \qquad&= \begin{pmatrix}
    y_1^T &  y_2^T
  \end{pmatrix}
            \begin{pmatrix}
              0 & 0\\
              0 & A_{22} - A_{21} (A_{11})^{-1}A_{12}
            \end{pmatrix}
                        \begin{pmatrix}
                          y_1\\ y_2
                        \end{pmatrix}\\
  \qquad&= \langle y_2,[(A^{-1})_{22}]^{-1} y_2\rangle
\end{aligned}
```

Because using the maximum-likelihood estimates ``\hat\beta`` has the effect of projecting out the fixed effects from the response ``y`` in the residual variance, it is common to also remove the contribution of the fixed effect space from the determinant term in the log-likelihood, i.e. replace

```math
\begin{aligned}
  \log\det(K+\delta I) = -\log\det\bigl[(K+\delta I)^{-1}\bigr] = -\log\det(A) 
  \to \log\det([A^{-1}]_{22}) = \log\det\bigl[K_{22}+\delta I\bigr],
\end{aligned}
```

 and consider the residual or restricted negative log-likelihood function

```math
  \mathcal{L}_R(\sigma^2,\delta) = \log\det\bigl[\sigma^2(K_{22}+\delta I)\bigr] + \frac1{\sigma^2} \langle y_2,(K_{22}+\delta I)^{-1} y_2\rangle.
```

This results in the restricted  maximum-likelihood estimate

```math
  \hat\sigma^2 = \frac1{n-d} \langle y_2,(K_{22}+\delta I)^{-1} y_2\rangle,
``` 

which is ``n/(n-d)`` times the unrestricted maximum-likelihood estimate. Plugging this in the restricted negative log-likelihood function gives, upto an additive constant

```math
  \mathcal{L}_R(\delta) = \log\det\bigl(K_{22}+\delta I\bigr) + (n-d)\log \hat\sigma^2.
```

Because the sample size ``n`` may be large, we scale this function by ``n-d`` to obtain

```math
\ell_R(\delta) = \frac1{n-d} \mathcal{L}_R(\delta) = \frac1{n-d}\log\det\bigl(K_{22}+\delta I\bigr) + \log \hat\sigma^2,
```

which needs to be minimized to obtain the restricted maximum-likelihood estimate ``\hat\delta``.

If ``K`` is given as a square matrix, the projection of ``y`` and ``K`` onto the space orthogonal to the columns of ``X`` is done by the function:

```@docs
svd_fixed_effects
```

## Fixed effects weights

The maximum-likelihood estimate ``\hat\beta`` for the fixed effects weights (see [Introduction](@ref)) can also be expressed conveniently using the same block matrix notation. Still writing ``A=(K+\delta I)^{-1}`` we have

```math
\begin{aligned}
\hat \beta &= (X^TAX)^{-1} X^T A y \\
&= W \Gamma^{-1} (A_{11})^{-1} \Gamma^{-1} W^T W \Gamma V_1^T A y\\
&= W \Gamma^{-1} (A_{11})^{-1} (Ay)_1\\
&= W \Gamma^{-1} (A_{11})^{-1} (A_{11}y_1 + A_{12}y_2)\\
&= W \Gamma^{-1} ( y_1 + (A_{11})^{-1} A_{12} y_2 )
\end{aligned}
```

Again using properties of the [inverse](https://en.wikipedia.org/wiki/Block_matrix#Inversion) of a [block matrix](https://en.wikipedia.org/wiki/Block_matrix), we have ``(A_{11})^{-1} A_{12} = - B_{12} (B_{22})^{-1}`` where ``B=A^{-1}=K+\delta I``, or

```math
\hat \beta = W \Gamma^{-1} ( y_1 - K_{12} (K_{22}+\delta I)^{-1} y_2 )
```