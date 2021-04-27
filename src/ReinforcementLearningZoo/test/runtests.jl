using ReinforcementLearningZoo
using Test
using ReinforcementLearningBase
using ReinforcementLearningCore
using ReinforcementLearningEnvironments
using Flux
using Statistics
using Random
using OpenSpiel
using StableRNGs
import GridWorlds

function get_optimal_kuhn_policy(α = 0.2)
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

@testset "ReinforcementLearningZoo.jl" begin

    @testset "training" begin
        mktempdir() do dir
            for method in (:BasicDQN, :BC, :DQN, :PrioritizedDQN, :Rainbow, :QRDQN, :REMDQN, :IQN, :VPG)
                res = run(
                    Experiment(
                        Val(:JuliaRL),
                        Val(method),
                        Val(:CartPole),
                        nothing;
                        save_dir = joinpath(dir, "CartPole", string(method)),
                    ),
                )
                @info "stats for $method" avg_reward = mean(res.hook[1].rewards) avg_fps =
                    1 / mean(res.hook[2].times)
            end

            for method in (:BasicDQN, :DQN)
                res = run(
                    Experiment(
                        Val(:JuliaRL),
                        Val(method),
                        Val(:MountainCar),
                        nothing;
                        save_dir = joinpath(dir, "MountainCar", string(method)),
                    ),
                )
                @info "stats for $method" avg_reward = mean(res.hook[1].rewards) avg_fps =
                    1 / mean(res.hook[2].times)
            end

            for method in (:A2C, :A2CGAE, :PPO, :MAC)
                res = run(
                    Experiment(
                        Val(:JuliaRL),
                        Val(method),
                        Val(:CartPole),
                        nothing;
                        save_dir = joinpath(dir, "CartPole", string(method)),
                    ),
                )
                @info "stats for $method" avg_reward =
                    mean(Iterators.flatten(res.hook[1].rewards))
            end

            for method in (:DDPG, :SAC, :TD3, :PPO)
                res = run(
                    Experiment(
                        Val(:JuliaRL),
                        Val(method),
                        Val(:Pendulum),
                        nothing;
                        save_dir = joinpath(dir, "Pendulum", string(method)),
                    ),
                )
                @info "stats for $method" avg_reward =
                    mean(Iterators.flatten(res.hook[1].rewards))
            end
        end
    end

    @testset "minimax" begin
        e = E`JuliaRL_Minimax_OpenSpiel(tic_tac_toe)`
        run(e)
        @test e.hook[1][] == e.hook[0][] == [0.0]
    end

    @testset "GridWorlds" begin
        mktempdir() do dir
            for method in (:BasicDQN,)
                res = run(
                    Experiment(
                        Val(:JuliaRL),
                        Val(method),
                        Val(:EmptyRoom),
                        nothing;
                        save_dir = joinpath(dir, "EmptyRoom", string(method)),
                    ),
                )
                @info "stats for $method" avg_reward = mean(res.hook[1].rewards) avg_fps =
                    1 / mean(res.hook[2].times)
            end
        end
    end

    @testset "TabularCFR" begin
        e = E`JuliaRL_TabularCFR_OpenSpiel(kuhn_poker)`
        run(e)

        reset!(e.env)
        expected_values = expected_policy_values(e.policy, e.env)
        @test isapprox(expected_values[1], -1 / 18; atol = 0.001)
        @test isapprox(expected_values[2], 1 / 18; atol = 0.001)
    end

    include("cfr/cfr.jl")
end
