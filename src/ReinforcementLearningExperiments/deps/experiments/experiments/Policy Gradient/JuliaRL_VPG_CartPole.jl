# ---
# title: JuliaRL\_VPG\_CartPole
# cover: assets/JuliaRL_VPG_CartPole.png
# description: VPG applied to CartPole
# date: 2022-07-17
# author: "[norci](https://github.com/norci)"
# ---

#+ tangle=true
using ReinforcementLearningCore
using ReinforcementLearningEnvironments
using StableRNGs: StableRNG
using Flux
using Flux: glorot_uniform
using Distributions: Categorical

function RLCore.Experiment(
    ::Val{:JuliaRL},
    ::Val{:VPG},
    ::Val{:CartPole};
    seed=123
)
    rng = StableRNG(seed)
    env = CartPoleEnv(; T=Float32, rng=rng)
    ns, na = length(state(env)), length(action_space(env))

    agent = Agent(
        policy=VPG(
            approximator=Approximator(
                model=Chain(
                    Dense(ns, 128, relu; init=glorot_uniform(rng)),
                    Dense(128, 128, relu; init=glorot_uniform(rng)),
                    Dense(128, na; init=glorot_uniform(rng)),
                ),
                optimiser=Adam(),
            ),
            baseline=Approximator(
                model=Chain(
                    Dense(ns, 128, relu; init=glorot_uniform(rng)),
                    Dense(128, 128, relu; init=glorot_uniform(rng)),
                    Dense(128, 1; init=glorot_uniform(rng)),
                ),
                optimiser=Adam(),
            ),
            dist=Categorical,
            γ=0.99f0,
            rng=rng,
        ),
        trajectory=Trajectory(container=Episode(ElasticArraySARTTraces(state=Float32 => (ns,))))
    )
    stop_condition = StopAfterEpisode(500, is_show_progress=!haskey(ENV, "CI"))

    hook = TotalRewardPerEpisode()
    Experiment(agent, env, stop_condition, hook)
end

#+ tangle=false
using Plots
pyplot() #hide
ex = E`JuliaRL_VPG_CartPole`
run(ex)
plot(ex.hook.rewards)
savefig("assets/JuliaRL_VPG_CartPole.png") #hide

# ![](assets/JuliaRL_VPG_CartPole.png)
