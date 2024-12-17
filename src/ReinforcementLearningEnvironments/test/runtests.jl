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
using JLD2

ENV["CONDA_JL_USE_MINIFORGE"] = "1"

Conda.add("python", Conda.ROOTENV)
Conda.add("numpy", Conda.ROOTENV)
Conda.pip_interop(true, Conda.ROOTENV)
Conda.pip("install", "gym", Conda.ROOTENV)


@testset "ReinforcementLearningEnvironments" begin
    include("environments/environments.jl")
end
