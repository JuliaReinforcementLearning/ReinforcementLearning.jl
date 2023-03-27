using Test
using MultiAgentReinforcementLearning
using StableRNGs
using PyCall
using ReinforcementLearningEnvironments
using ReinforcementLearningBase
using ReinforcementLearningCore
using ReinforcementLearningZoo
using Distributions
using Flux
using Flux: glorot_uniform


@testset "MultiAgentReinforcementLearning" begin
    include("independent_learner.jl")
    include("maddpg_learner.jl")
    include("tic_tac_toe.jl")
end
