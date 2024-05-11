using FaSTLMMlight
using Test

using LinearAlgebra
using Statistics

@testset "FaSTLMMlight.jl" begin
    @testset "FaST-LMM core" begin
        include("fastlmm-core_tests.jl")
    end
end
