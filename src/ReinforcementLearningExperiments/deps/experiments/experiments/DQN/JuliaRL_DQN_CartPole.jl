
# ---
# title: JuliaRL\_BasicDQN\_CartPole
# cover: assets/JuliaRL_BasicDQN_CartPole.png
# description: The simplest example to demonstrate how to use BasicDQN
# date: 2022-06-04
# author: "[Jun Tian](https://github.com/findmyway)"
# ---


using ReinforcementLearningCore, ReinforcementLearningBase, ReinforcementLearningZoo
using ReinforcementLearningEnvironments
using Flux
using Flux: glorot_uniform

using StableRNGs: StableRNG
using Flux.Losses: huber_loss

using Flux.Losses: huber_loss
using Zygote
using Statistics: mean

function RLCore.Experiment(
    ::Val{:JuliaRL},
    ::Val{:BasicDQN},
    ::Val{:CartPole};
    seed=123
)
    rng = StableRNG(seed)
    env = CartPoleEnv(; T=Float32, rng=rng)
    ns, na = length(state(env)), length(action_space(env))

    agent = Agent(
        policy=QBasedPolicy(
            learner=DQNLearner(
                approximator=Approximator(
                    model=Chain(
                        Dense(ns, 128, relu; init=glorot_uniform(rng)),
                        Dense(128, 128, relu; init=glorot_uniform(rng)),
                        Dense(128, na; init=glorot_uniform(rng)),
                    ),
                    optimiser=Adam(),
                ),
                loss_func=mse, # TODO: revert to huber_loss when https://github.com/JuliaGPU/GPUArrays.jl/issues/484 is fixed
            ),
            explorer=EpsilonGreedyExplorer(
                kind=:exp,
                ϵ_stable=0.01,
                decay_steps=500,
                rng=rng,
            ),
        ),
        trajectory=Trajectory(
            container=CircularArraySARTSTraces(
                capacity=1000,
                state=Float32 => (ns,),
            ),
            sampler=BatchSampler{SS′ART}(
                batch_size=32,
                rng=rng
            ),
            controller=InsertSampleRatioController(
                threshold=100,
                n_inserted=-1
            )
        )
    )
    stop_condition = StopAfterStep(10_000, is_show_progress=!haskey(ENV, "CI"))
    hook = TotalRewardPerEpisode()
    Experiment(agent, env, stop_condition, hook)
end
