using Test
using ReinforcementLearningBase
using ReinforcementLearningEnvironments
using ArcadeLearningEnvironment
using PyCall
using POMDPs
using POMDPModels
using OpenSpiel
using Random

RLBase.get_observation_space(m::TigerPOMDP) = DiscreteSpace((false, true))

@testset "ReinforcementLearningEnvironments" begin

    include("environments.jl")
    include("atari.jl")
    include("open_spiel.jl")
end
