
# ---
# title: JuliaRL\_DQN\_CartPole
# cover: assets/JuliaRL_DQN_CartPole.png
# description: DQN applied to CartPole
# date: 2022-06-06
# author: "[Jun Tian](https://github.com/findmyway)"
# ---


using ReinforcementLearning
using Flux
using Flux: glorot_uniform

using StableRNGs: StableRNG
using Flux.Losses: huber_loss

function RL.Experiment(
    ::Val{:JuliaRL},
    ::Val{:DQN},
    ::Val{:CartPole};
    seed=123
)
    rng = StableRNG(seed)
    env = CartPoleEnv(; T=Float32, rng=rng)
    ns, na = length(state(env)), length(action_space(env))

    agent = Agent(
        policy=QBasedPolicy(
            learner=DQNLearner(
                approximator=NeuralNetworkApproximator(
                    model=DuelingNetwork(
                        base=Chain(
                            Dense(ns, 128, relu; init=glorot_uniform(rng)),
                            Dense(128, 128, relu; init=glorot_uniform(rng)),
                        ),
                        val=Dense(128, 1; init=glorot_uniform(rng)),
                        adv=Dense(128, na; init=glorot_uniform(rng)),
                    ),
                    optimizer=ADAM(),
                ) |> gpu,
                target_approximator=NeuralNetworkApproximator(
                    model=DuelingNetwork(
                        base=Chain(
                            Dense(ns, 128, relu; init=glorot_uniform(rng)),
                            Dense(128, 128, relu; init=glorot_uniform(rng)),
                        ),
                        val=Dense(128, 1; init=glorot_uniform(rng)),
                        adv=Dense(128, na; init=glorot_uniform(rng)),
                    ),
                ) |> gpu,
                loss_func=huber_loss,
                stack_size=nothing,
                batch_size=32,
                update_horizon=1,
                min_replay_history=100,
                update_freq=1,
                target_update_freq=100,
                rng=rng,
            ),
            explorer=EpsilonGreedyExplorer(
                kind=:exp,
                ϵ_stable=0.01,
                decay_steps=500,
                rng=rng,
            ),
        ),
        trajectory=CircularArraySARTTrajectory(
            capacity=1000,
            state=Vector{Float32} => (ns,),
        ),
    )

    stop_condition = StopAfterStep(10_000, is_show_progress=!haskey(ENV, "CI"))
    hook = TotalRewardPerEpisode()
    Experiment(agent, env, stop_condition, hook, "")
end

#+ tangle=false
using Plots
pyplot() #hide
ex = E`JuliaRL_DQN_CartPole`
run(ex)
plot(ex.hook.rewards)
savefig("assets/JuliaRL_DQN_CartPole.png") #hide

# ![](assets/JuliaRL_DQN_CartPole.png)
