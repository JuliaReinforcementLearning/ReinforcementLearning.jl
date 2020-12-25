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

@testset "ReinforcementLearningEnvironments" begin
    include("environments/environments.jl")
end
