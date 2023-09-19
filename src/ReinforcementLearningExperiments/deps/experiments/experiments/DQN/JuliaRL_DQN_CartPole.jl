# ---
# title: JuliaRL\_DQN\_CartPole
# cover: assets/JuliaRL_DQN_CartPole.png
# description: DQN applied to CartPole
# date: 2022-06-12
# author: "[Jun Tian](https://github.com/findmyway)"
# ---


using ReinforcementLearningCore, ReinforcementLearningBase, ReinforcementLearningZoo
using ReinforcementLearningEnvironments
using Flux
using Flux: glorot_uniform

using StableRNGs: StableRNG
using Flux.Losses: huber_loss

function RLCore.Experiment(
    ::Val{:JuliaRL},
    ::Val{:DQN},
    ::Val{:CartPole};
    seed=123,
    n=1,
    γ=0.99f0,
)
    rng = StableRNG(seed)
    env = CartPoleEnv(; T=Float32, rng=rng)
    ns, na = length(state(env)), length(action_space(env))

    agent = Agent(
        policy=QBasedPolicy(
            learner=DQNLearner(
                approximator=TargetNetwork(
                    Approximator(
                        model = Chain(
                            Dense(ns, 128, relu; init=glorot_uniform(rng)),
                            Dense(128, 128, relu; init=glorot_uniform(rng)),
                            Dense(128, na; init=glorot_uniform(rng)),
                            ),
                        optimiser=Adam()
                        ),
                    sync_freq=100
                ),
                n=n,
                γ=γ,
                loss_func=huber_loss,
                rng=rng,
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
            sampler=NStepBatchSampler{SS′ART}(
                n=n,
                γ=γ,
                batchsize =32,
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

#+ tangle=false
using Plots
pyplot() #hide
ex = E`JuliaRL_DQN_CartPole`
run(ex)
plot(ex.hook.rewards)
savefig("assets/JuliaRL_DQN_CartPole.png") #hide

# ![](assets/JuliaRL_DQN_CartPole.png)
