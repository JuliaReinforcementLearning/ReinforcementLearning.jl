using ReinforcementLearningZoo
using Test
using ReinforcementLearningBase
using ReinforcementLearningCore
using ReinforcementLearningEnvironments
using Flux
using Statistics
using Random

@testset "ReinforcementLearningZoo.jl" begin

    @testset "training" begin
        mktempdir() do dir
            for method in (:BasicDQN, :DQN, :PrioritizedDQN, :Rainbow, :IQN)
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

            for method in (:DDPG, :SAC)
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
        for x in ("JuliaRL_BasicDQN_CartPole",)
            e = Experiment(x)
            e.agent.policy = load_policy(x)
            Flux.testmode!(e.agent)
            run(e.agent, e.env, StopAfterEpisode(1), e.hook)
            @info "result of evaluating pretrained model: $x for once:" reward =
                e.hook[1].rewards[end]
        end
    end
end
