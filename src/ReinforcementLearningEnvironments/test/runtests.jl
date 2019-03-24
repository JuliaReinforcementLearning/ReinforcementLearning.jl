using Test
using ReinforcementLearningEnvironments
using ArcadeLearningEnvironment
using POMDPModels
using ViZDoom
using PyCall

@testset "ReinforcementLearningEnvironments" begin

include("spaces.jl")
include("environments.jl")

end