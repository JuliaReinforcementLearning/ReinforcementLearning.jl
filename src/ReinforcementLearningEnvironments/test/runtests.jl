using Test
using ReinforcementLearningBase
using ReinforcementLearningCore
using ReinforcementLearningEnvironments
using ArcadeLearningEnvironment
using PyCall
using OpenSpiel
# using SnakeGames
using Random
using StableRNGs
using Statistics
using OrdinaryDiffEq
using TimerOutputs
using Conda

Conda.add("python")
Conda.add("gym")
Conda.add("numpy")

@testset "ReinforcementLearningEnvironments" begin
    include("environments/environments.jl")
end
