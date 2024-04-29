using FaSTLMMlight
using Test

using LinearAlgebra
using Statistics

@testset "FaSTLMMlight.jl" begin
    @testset "FaST-LMM fullrank" begin
        include("fastlmm-fullrank_tests.jl")
    end
end
