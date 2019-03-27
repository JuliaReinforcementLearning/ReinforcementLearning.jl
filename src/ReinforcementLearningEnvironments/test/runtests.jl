using Test
using ReinforcementLearningEnvironments
using ArcadeLearningEnvironment
using POMDPModels
using ViZDoom
using PyCall
using Hanabi

@testset "ReinforcementLearningEnvironments" begin

include("spaces.jl")
include("environments.jl")

end