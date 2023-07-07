# ---
# title: JuliaRL\_REMDQN\_CartPole
# cover: assets/JuliaRL_REMDQN_CartPole.png
# description: REMDQN applied to CartPole
# date: 2021-06-25
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
    ::Val{:REMDQN},
    ::Val{:CartPole},
    ; seed=123,
    ensemble_num=16
)
    rng = StableRNG(seed)

    env = CartPoleEnv(; T=Float32, rng=rng)
    ns, na = length(state(env)), length(action_space(env))

    n = 1
    γ = 0.99f0

    agent = Agent(
        policy=QBasedPolicy(
            learner=REMDQNLearner(
                approximator=Approximator(
                    model=TwinNetwork(
                        Chain(
                            ## Multi-head method, please refer to "https://github.com/google-research/batch_rl/tree/b55ba35ebd2381199125dd77bfac9e9c59a64d74/batch_rl/multi_head".
                            Dense(ns, 128, relu; init=glorot_uniform(rng)),
                            Dense(128, 128, relu; init=glorot_uniform(rng)),
                            Dense(128, na * ensemble_num; init=glorot_uniform(rng)),
                        ),
                        sync_freq=100
                    ),
                    optimiser=Adam(),
                ),
                n=n,
                γ=γ,
                loss_func=huber_loss,
                ensemble_num=ensemble_num,
                ensemble_method=:rand,
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
            container=CircularArraySARSTTraces(
                capacity=1000,
                state=Float32 => (ns,),
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

    stop_condition = StopAfterStep(10_000, is_show_progress=!haskey(ENV, "CI"))
    hook = TotalRewardPerEpisode()

    ## !!! note that REMDQN is used in offline RL
    ## TODO: use DQN to collect experiences and then optimise the REMDQN
    Experiment(agent, env, stop_condition, hook)
end

#+ tangle=false
using Plots
pyplot() #hide
ex = E`JuliaRL_REMDQN_CartPole`
run(ex)
plot(ex.hook.rewards)
savefig("assets/JuliaRL_REMDQN_CartPole.png") #hide

# ![](assets/JuliaRL_REMDQN_CartPole.png)
