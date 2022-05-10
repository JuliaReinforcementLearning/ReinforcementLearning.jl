using CircularArrayBuffers
using ReinforcementLearningBase
using ReinforcementLearningCore
using Random
using Test
using StatsBase
using Distributions: probs, Normal, logpdf, MvNormal
using ReinforcementLearningEnvironments
using Flux
using CUDA
using LinearAlgebra

@testset "ReinforcementLearningCore.jl" begin
    include("core.jl")
    include("utils/utils.jl")
end
