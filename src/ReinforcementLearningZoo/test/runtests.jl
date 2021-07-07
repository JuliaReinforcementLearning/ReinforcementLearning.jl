using ReinforcementLearningZoo
using Test
using ReinforcementLearningBase
using ReinforcementLearningCore
using ReinforcementLearningEnvironments
using Flux
using Statistics
using Random
using StableRNGs
using OpenSpiel

function get_optimal_kuhn_policy(env, α = 0.2)
    if typeof(env) == KuhnPokerEnv
        TabularRandomPolicy(
            table = Dict(
                (:J,) => [1 - α, α],
                (:J, :pass, :bet) => [1.0, 0.0],
                (:Q,) => [1.0, 0.0],
                (:Q, :pass, :bet) => [2.0 / 3.0 - α, 1.0 / 3.0 + α],
                (:K,) => [1 - 3 * α, 3 * α],
                (:K, :pass, :bet) => [0.0, 1.0],
                (:J, :pass) => [2.0 / 3.0, 1.0 / 3.0],
                (:J, :bet) => [1.0, 0.0],
                (:Q, :pass) => [1.0, 0.0],
                (:Q, :bet) => [2.0 / 3.0, 1.0 / 3.0],
                (:K, :pass) => [0.0, 1.0],
                (:K, :bet) => [0.0, 1.0],
            ),
        )

    else
        TabularRandomPolicy(
            table = Dict(
                "0" => [1 - α, α],
                "0pb" => [1.0, 0.0],
                "1" => [1.0, 0.0],
                "1pb" => [2.0 / 3.0 - α, 1.0 / 3.0 + α],
                "2" => [1 - 3 * α, 3 * α],
                "2pb" => [0.0, 1.0],
                "0p" => [2.0 / 3.0, 1.0 / 3.0],
                "0b" => [1.0, 0.0],
                "1p" => [1.0, 0.0],
                "1b" => [2.0 / 3.0, 1.0 / 3.0],
                "2p" => [0.0, 1.0],
                "2b" => [0.0, 1.0],
            ),
        )
    end
end

@testset "ReinforcementLearningZoo.jl" begin
    include("cfr/cfr.jl")
end
