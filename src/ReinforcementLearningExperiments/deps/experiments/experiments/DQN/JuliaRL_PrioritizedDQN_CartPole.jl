# ---
# title: JuliaRL\_PrioritizedDQN\_CartPole
# cover: assets/JuliaRL_PrioritizedDQN_CartPole.png
# description: PrioritizedDQN applied to CartPole
# date: 2022-06-18
# author: "[Jun Tian](https://github.com/findmyway)"
# ---

#+ tangle=true
using ReinforcementLearningCore, ReinforcementLearningBase, ReinforcementLearningZoo
using ReinforcementLearningEnvironments
using StableRNGs
using Flux
using Flux.Losses

function RLCore.Experiment(
    ::Val{:JuliaRL},
    ::Val{:PrioritizedDQN},
    ::Val{:CartPole},
    ; seed=123,
    n=1,
    γ=0.99f0
)
    rng = StableRNG(seed)

    env = CartPoleEnv(; T=Float32, rng=rng)
    ns, na = length(state(env)), length(action_space(env))

    agent = Agent(
        policy=QBasedPolicy(
            learner=PrioritizedDQNLearner(
                approximator=TargetNetwork(
                    Approximator(
                        model = Chain(
                            Dense(ns, 128, relu; init=Flux.glorot_uniform(rng)),
                            Dense(128, 128, relu; init=Flux.glorot_uniform(rng)),
                            Dense(128, na; init=Flux.glorot_uniform(rng)),
                        );
                        optimiser=Adam(),
                        ),
                    sync_freq=100
                ),
                n=n,
                γ=γ,
                β_priority=0.5f0,
                loss_func=(ŷ, y) -> huber_loss(ŷ, y; agg=identity),
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
            container=CircularPrioritizedTraces(
                CircularArraySARTSTraces(
                    capacity=1000,
                    state=Float32 => (ns,),
                );
                default_priority=100.0f0
            ),
            sampler=NStepBatchSampler{SS′ART}(
                n=n,
                γ=γ,
                batch_size=32,
                rng=rng
            ),
            controller=InsertSampleRatioController(
                threshold=100,
                n_inserted=-1
            )
        )
    )

    stop_condition = StopAfterStep(10_000)
    hook = TotalRewardPerEpisode()
    Experiment(agent, env, stop_condition, hook)
end


#+ tangle=false
using Plots
pyplot() #hide
ex = E`JuliaRL_PrioritizedDQN_CartPole`
run(ex)
plot(ex.hook.rewards)
savefig("assets/JuliaRL_PrioritizedDQN_CartPole.png") #hide

# ![](assets/JuliaRL_PrioritizedDQN_CartPole.png)
