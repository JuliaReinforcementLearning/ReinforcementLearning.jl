using ReinforcementLearningZoo
using Test
using ReinforcementLearningBase
using ReinforcementLearningCore
using ReinforcementLearningEnvironments
using Flux
using Statistics

@testset "ReinforcementLearningZoo.jl" begin
    mktempdir() do dir
        for method in (:BasicDQN, :DQN, :PrioritizedDQN, :Rainbow)
            res = run(Experiment(
                Val(:JuliaRL),
                Val(method),
                Val(:CartPole),
                nothing;
                save_dir = joinpath(dir, string(method)),
            ))
            @info "stats for $method" avg_reward = mean(res.hook[1].rewards) avg_fps =
                1 / mean(res.hook[2].times)
        end

        for method in (:A2C, :A2CGAE)
            res = run(Experiment(Val(:JuliaRL), Val(method), Val(:CartPole), nothing))
            @info "stats for $method" avg_reward = mean(Iterators.flatten(res.hook.rewards))
        end
    end
end
