module FaSTLMMlight

# dependencies
using LinearAlgebra
using Statistics
using Optim

# Code files
include("fastlmm-fullrank.jl")

# FaST-LMM Exports
export fastlmm_fullrank, sigma2_mle_fullrank, minus_log_like_fullrank, delta_mle_fullrank, beta_mle_fullrank_lazy, create_covariate_matrix, project_orth_covar, softplus

end
