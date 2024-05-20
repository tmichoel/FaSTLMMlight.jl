"""
    fastlmm_fullrank(y,K; covariates = [], mean = true)

For a linear mixed model / Gaussian process,

```math
y \\sim N\\bigl(X\\beta, \\sigma^2(K + \\delta I) \\bigr)
```

where ``y`` is the response vector, ``X`` is a matrix of covariates,  and ``K`` is  a full-rank kernel matrix, compute the restricted maximum-likelihood estimates (REMLs) of the variance parameter ``\\sigma^2`` and the variance ratio ``\\delta`` using FaST-LMM with a full-rank kernel matrix. Compared to the original FaST-LMM algorithm, we first project out the (optional) covariates, incl. an (optional) constant off-set (`mean=true`), from the response vector and the kernel matrix. This avoids all matrix computations in the variance parameter estimation. Estimates for the fixed effects ``\\beta`` are not computed.
"""
function fastlmm_fullrank(y,K; covariates = [], mean = true, lambda_tol = 1e-3)
    # Create covariate matrix X from the provided covariates with an intercept column if mean=true
    X = create_covariate_matrix(covariates; mean = mean, n = size(y,1))
    
    # if X is not empty, project it out from the response vector and the kernel matrix
    if !isempty(X)
        y1, y2, K12, K22, γ, Wt  = svd_fixed_effects(y, K, X)
    end
    
    # Eigendecomposition of the kernel matrix
    EF = eigen(K22);
    λ = EF.values;
    U = EF.vectors;

    # Due to numerical issues, we set eigenvalues smaller than lambda_tol to zero
    λ[λ .< lambda_tol] .= 0.0

    # Rotate the data
    yr = U' * y2;

    if size(yr,2) == 1
        # Compute the REML of the variance ratio δ
        δ, res = delta_reml(λ, yr);
        # Compute the REML of the variance parameter σ²
        σ² = sigma2_reml(δ, λ, yr);
        # Compute the MLE of the fixed effects weights
        β = beta_mle(δ, λ, yr, y1, K12, U, γ, Wt)
        # return the REMLs of the variance parameter, the MLE of the fixed effects weights and the full optimization result
        return σ², δ, β, res
    else 
        # Compute and return the REMLs of the variance ratio δ and variance parameter σ² for each column of yr
        δs = zeros(size(yr,2));
        σ²s = zeros(size(yr,2));
        loglikes = zeros(size(yr,2));
        βs = zeros(size(yr,2), size(X,2));
        for i in eachindex(axes(yr)[2])
            δs[i], res = delta_reml(λ, yr[:,i]);
            σ²s[i] = sigma2_reml(δs[i], λ, yr[:,i]);
            βs[i,:] = beta_mle(δs[i], λ, yr[:,i], y1[:,i], K12, U, γ, Wt)
            loglikes[i] = minimum(res)
        end
        # return the MLEs and the final objective values (minus log-likelihoods)
        return σ²s, δs, βs, loglikes
    end
end

"""
    delta_reml(λ, yr)

Compute the REML of the variance ratio δ given the eigenvalues of the kernel matrix and the rotated response vector by solving the non-convex optimization problem formulated in the FaST-LMM paper.
"""
function delta_reml(λ, yr)
    # Compute the MLE of the variance parameter
    res = Optim.optimize(x -> neg_log_like(softplus(x), λ, yr), [0.0], LBFGS(); autodiff = :forward);
    xmin = res.minimizer;
    return softplus(xmin[1]), res
end

"""
    neg_log_like(δ, λ, yr)

Compute the negative (restircted) log-likelihood of the model, scaled by the number of samples and without constant factors, given the variance ratio δ, the eigenvalues of the kernel matrix, and the rotated response vector.
"""
function neg_log_like(δ, λ, yr)
    # Compute the minus log-likelihood of the model, scaled by half the number of samples and without constant factors
    σ² = sigma2_reml(δ, λ, yr);
    return  mean(log.(abs.(λ .+ δ))) .+ log(σ²)
end

"""
    sigma2_reml(δ, λ, yr)

Compute the REML of the variance parameter given the variance ratio δ, the eigenvalues of the kernel matrix, and the rotated response vector.
"""
function sigma2_reml(δ, λ, yr)
    # Compute the MLE of the variance parameter
    return mean(yr.^2 ./ (λ .+ δ))
end


"""
    beta_mle(δ, λ, yr, y1, K12, U, γ, Wt)

Compute the MLE of the fixed effects weights given the variance ratio δ, the eigenvalues of the kernel matrix, the rotated response vector, the rotated response vector projected onto the orthogonal complement of the column space of the covariates, the block decomposition of the kernel matrix, the eigenvectors of the kernel matrix, and the eigenvalues of the kernel matrix.
"""
function beta_mle(δ, λ, yr, y1, K12, U, γ, Wt)
    β = Wt * ( (y1 - K12 * U * (yr ./ (λ .+ δ))) ./ γ )
    if length(β) == 1
        return β[1]
    else
        return β
    end
end

# """
#     beta_mle_fullrank(δ, λ, yr, Xr=[])

# Compute the MLE of the fixed effects weights given the variance ratio δ, the eigenvalues of the kernel matrix, the rotated response vector and (optionally) the rotated covariates.
# """
# function beta_mle_fullrank(δ, λ, yr, Xr=[])
#     # Compute the MLE of the fixed effects weights
#     if isempty(Xr)
#         return 0.0 # no covariates
#     elseif size(Xr,2) == 1
#         return sum(Xr .* yr ./ (λ .+ δ)) / sum(Xr.^2 ./ (λ .+ δ)) 
#     else
#         A = Xr' * diagm(1 ./ (λ .+ δ)) * Xr
#         b = Xr' * (yr ./ (λ .+ δ))
#         return A \ b
#     end  
# end



"""
    beta_mle_fullrank_lazy(y, K, X, σ², δ)

Lazy implementation of the MLE of the fixed effects weights given the response vector `y`, the kernel matrix `K`, the covariates `X`, the variance parameter `σ²` and the variance ratio `δ`. This function does not use spectral factorization, and should only be applied once, using the MLEs of the variance parameters.
"""
function beta_mle_fullrank_lazy(y, K, X, σ², δ; mean = true)
    # Create the covariance matrix
    if isempty(K)
        Σ = σ² * I
    else
        Σ = σ² * (K + δ*I)
    end
    # Create the covariate matrix
    X = create_covariate_matrix(X; mean = mean, n = size(y,1))
    # Compute the MLE of the fixed effects weights
    if isempty(X)
        return 0.0 # no covariates
    elseif size(X,2) == 1
        return dot(X, Σ \ y) / dot(X, Σ \ X)
    else
        A = X' * (Σ \ X)
        b = X' * (Σ \ y) 
        return A \ b
    end  
end


"""
    create_covariate_matrix(X; mean = true, n = 1)

Create the covariate matrix from the provided covariates with an intercept column if `mean=true`.
"""
function create_covariate_matrix(X; mean = true, n = 1)
    if !isempty(X)
        n = size(X,1)
        if mean
            X = hcat(ones(n), X)
        end
    elseif mean
        X = ones(n)
    else
        X = []  
    end
    return X
end

"""
    svd_fixed_effects(y, K, X)

Compute the SVD of the covariate matrix `X` and return the block deomposition of the response vector `y` and the kernel matrix `K` with respect to the orthogonal complement of the column space of `X`.
"""
function svd_fixed_effects(y, K, X)
    # SVD of the covariate matrix
    F  = svd(X, full=true);
    # Get the vectors that span the range and orthogonal complement of the column space of X
    V1 = F.U[:,1:length(F.S)];
    V2 = F.U[:,length(F.S)+1:end];
    # Compute the block decomposition of the response vector and the kernel matrix
    y1 = V1' * y;
    y2 = V2' * y;
    if !isempty(K)
        K12 = V1' * K * V2;
        K22 = V2' * K * V2;
        # make sure K2 is symmetric
        K22 = (K22 + K22') / 2
    else
        K12 = []
        K22 = []
    end
    return y1, y2, K12, K22, F.S, F.Vt
end

"""
    softplus(x)

Compute the softplus function, i.e. log(1 + exp(x)), element-wise.
"""
function softplus(x)
    return log.(1 .+ exp.(x))
end

