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

function get_optimal_kuhn_policy(α=0.2)
    TabularRandomPolicy(table=Dict(
        "0"   => [1 - α, α],
        "0pb" => [1., 0.],
        "1"   => [1., 0.],
        "1pb" => [2. / 3. - α, 1. / 3. + α],
        "2"   => [1 - 3 * α, 3 * α],
        "2pb" => [0., 1.],

        "0p"  => [2. / 3., 1. / 3.],
        "0b"  => [1., 0.],
        "1p"  => [1., 0.],
        "1b"  => [2. / 3., 1. / 3.],
        "2p"  => [0., 1.],
        "2b"  => [0., 1.],
    ))
end

@testset "ReinforcementLearningZoo.jl" begin

    @testset "training" begin
        mktempdir() do dir
            for method in (:BasicDQN, :DQN, :PrioritizedDQN, :Rainbow, :IQN, :VPG)
                res = run(Experiment(
                    Val(:JuliaRL),
                    Val(method),
                    Val(:CartPole),
                    nothing;
                    save_dir = joinpath(dir, "CartPole", string(method)),
                ))
                @info "stats for $method" avg_reward = mean(res.hook[1].rewards) avg_fps =
                    1 / mean(res.hook[2].times)
            end

            for method in (:BasicDQN, :DQN)
                res = run(Experiment(
                    Val(:JuliaRL),
                    Val(method),
                    Val(:MountainCar),
                    nothing;
                    save_dir = joinpath(dir, "MountainCar", string(method)),
                ))
                @info "stats for $method" avg_reward = mean(res.hook[1].rewards) avg_fps =
                    1 / mean(res.hook[2].times)
            end

            for method in (:A2C, :A2CGAE, :PPO)
                res = run(Experiment(
                    Val(:JuliaRL),
                    Val(method),
                    Val(:CartPole),
                    nothing;
                    save_dir = joinpath(dir, "CartPole", string(method)),
                ))
                @info "stats for $method" avg_reward =
                    mean(Iterators.flatten(res.hook[1].rewards))
            end

            for method in (:DDPG, :SAC, :TD3)
                res = run(Experiment(
                    Val(:JuliaRL),
                    Val(method),
                    Val(:Pendulum),
                    nothing;
                    save_dir = joinpath(dir, "Pendulum", string(method)),
                ))
                @info "stats for $method" avg_reward =
                    mean(Iterators.flatten(res.hook[1].rewards))
            end
        end
    end

    @testset "run pretrained models" begin
        #=
        # the old pretrained model can not be loaded
        # TODO: how to handle version conflict of pretrained models?
        for x in ("JuliaRL_BasicDQN_CartPole",)
            e = Experiment(x)
            e.agent.policy = load_policy(x)
            Flux.testmode!(e.agent)
            run(e.agent, e.env, StopAfterEpisode(1), e.hook)
            @info "result of evaluating pretrained model: $x for once:" reward =
                e.hook[1].rewards[end]
        end
        =#
    end

    @testset "minimax" begin
        e = E`JuliaRL_Minimax_OpenSpiel(tic_tac_toe)`
        run(e)
        @test e.hook[1].rewards[end] == e.hook[2].rewards[end] == 0.0
    end

    @testset "TabularCFR" begin
        e = E`JuliaRL_TabularCFR_OpenSpiel(kuhn_poker)`
        run(e)

        reset!(e.env)
        expected_values = expected_policy_values(e.agent, e.env)
        @test isapprox(expected_values[1], -1 / 18; atol = 0.001)
        @test isapprox(expected_values[2], 1 / 18; atol = 0.001)
    end

    include("cfr/cfr.jl")
end
