using Test
using ReinforcementLearningBase
using ReinforcementLearningEnvironments
using ArcadeLearningEnvironment
using PyCall
using OpenSpiel
# using SnakeGames
using Random
using StableRNGs
using Statistics
using OrdinaryDiffEq

@testset "ReinforcementLearningEnvironments" begin
    include("environments/environments.jl")
end
