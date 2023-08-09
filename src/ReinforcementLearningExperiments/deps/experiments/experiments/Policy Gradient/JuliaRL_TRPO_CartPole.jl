# ---
# title: JuliaRL\_TRPO\_CartPole
# cover: assets/JuliaRL_TRPO_CartPole.png
# description: TRPO applied to CartPole
# date: 2022-08-08
# author: "[baedan](https://github.com/baedan)"
# ---

#+ tangle=true
using ReinforcementLearningCore, ReinforcementLearningBase, ReinforcementLearningZoo
using ReinforcementLearningEnvironments
using Flux: Flux, glorot_uniform
using StableRNGs: StableRNG
using Distributions: Categorical

function RLCore.Experiment(
    ::Val{:JuliaRL},
    ::Val{:TRPO},
    ::Val{:CartPole};
    seed=123
)
    rng = StableRNG(seed)
    env = CartPoleEnv(; T=Float32, rng=rng)
    ns, na = length(state(env)), length(action_space(env))

    agent = Agent(
        policy=TRPO(
            approximator=Approximator(
                model=Chain(
                    Dense(ns, 64, relu; init=glorot_uniform(rng)),
                    Dense(64, 16, relu; init=glorot_uniform(rng)),
                    Dense(16, 4, relu; init=glorot_uniform(rng)),
                    Dense(4, na; init=glorot_uniform(rng)),
                ),
                optimiser=Descent(0.02),
            ),
            baseline=Approximator(
                model=Chain(
                    Dense(ns, 64, relu; init=glorot_uniform(rng)),
                    Dense(64, 16, relu; init=glorot_uniform(rng)),
                    Dense(16, 4, relu; init=glorot_uniform(rng)),
                    Dense(4, 1; init=glorot_uniform(rng)),
                ),
                optimiser=Descent(0.005),
            ),
            rng=rng,
        ),
        trajectory=Trajectory(container=CircularArraySARTTraces(capacity = 10000, state=Float32 => (ns,)), sampler = EpisodesSampler(), controller = InsertSampleRatioController(ratio = 1/10000))
    )
    stop_condition = StopAfterEpisode(100, is_show_progress=!haskey(ENV, "CI"))

    hook = TotalRewardPerEpisode()
    Experiment(agent, env, stop_condition, hook)
end

#+ tangle=false
using Plots
pyplot() #hide
ex = E`JuliaRL_TRPO_CartPole`
run(ex)
plot(ex.hook.rewards)
savefig("assets/JuliaRL_TRPO_CartPole.png") #hide

# ![](assets/JuliaRL_TRPO_CartPole.png)
