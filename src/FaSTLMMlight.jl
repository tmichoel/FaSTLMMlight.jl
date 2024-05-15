module FaSTLMMlight

# dependencies
using LinearAlgebra
using Statistics
using Optim

# Code files
include("fastlmm-core.jl")

# FaST-LMM Exports
export fastlmm_fullrank, sigma2_reml, neg_log_like, delta_reml, beta_mle, beta_mle_fullrank_lazy, create_covariate_matrix, svd_fixed_effects, softplus

end
