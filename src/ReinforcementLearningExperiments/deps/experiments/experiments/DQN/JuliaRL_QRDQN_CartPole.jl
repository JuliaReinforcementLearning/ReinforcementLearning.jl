# ---
# title: JuliaRL\_QRDQN\_CartPole
# cover: assets/JuliaRL_QRDQN_CartPole.png
# description: QRDQN applied to CartPole
# date: 2021-06-19
# author: "[Jun Tian](https://github.com/findmyway)"
# ---

#+ tangle=true
using ReinforcementLearningCore, ReinforcementLearningBase, ReinforcementLearningZoo
using ReinforcementLearningEnvironments
using StableRNGs: StableRNG
using Flux
using Flux: glorot_uniform

function RLCore.Experiment(
    ::Val{:JuliaRL},
    ::Val{:QRDQN},
    ::Val{:CartPole},
    ; seed=123
)

    N = 10

    rng = StableRNG(seed)
    env = CartPoleEnv(; T=Float32, rng=rng)
    ns, na = length(state(env)), length(action_space(env))

    agent = Agent(
        policy=QBasedPolicy(
            learner=QRDQNLearner(
                approximator=Approximator(
                    model=TwinNetwork(
                        Chain(
                            Dense(ns, 128, relu; init=glorot_uniform(rng)),
                            Dense(128, 128, relu; init=glorot_uniform(rng)),
                            Dense(128, N * na; init=glorot_uniform(rng)),
                        );
                        sync_freq=100
                    ),
                    optimiser=Adam(),
                ),
                n_quantile=N,
                loss_func=quantile_huber_loss,
                γ=0.99f0,
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
            container=CircularArraySARTTraces(
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

#+ tangle=false
using Plots
pyplot() #hide
ex = E`JuliaRL_QRDQN_CartPole`
run(ex)
plot(ex.hook.rewards)
savefig("assets/JuliaRL_QRDQN_CartPole.png") #hide

# ![](assets/JuliaRL_QRDQN_CartPole.png)
