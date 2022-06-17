# ---
# title: JuliaRL\_QRDQN\_CartPole
# cover: assets/JuliaRL_QRDQN_CartPole.png
# description: QRDQN applied to CartPole
# date: 2021-05-22
# author: "[Jun Tian](https://github.com/findmyway)"
# ---

#+ tangle=true
using ReinforcementLearning
using StableRNGs
using Flux
using Flux.Losses

function RL.Experiment(
    ::Val{:JuliaRL},
    ::Val{:QRDQN},
    ::Val{:CartPole},
    ::Nothing;
    seed=123,
)

    N = 10

    rng = StableRNG(seed)
    env = CartPoleEnv(; T=Float32, rng=rng)
    ns, na = length(state(env)), length(action_space(env))

    init = glorot_uniform(rng)

    agent = Agent(
        policy=QBasedPolicy(
            learner=QRDQNLearner(
                approximator=NeuralNetworkApproximator(
                    model=Chain(
                        Dense(ns, 128, relu; init = init),
                        Dense(128, 128, relu; init = init),
                        Dense(128, N * na; init = init),
                    ) |> gpu,
                    optimizer=ADAM(),
                ),
                target_approximator=NeuralNetworkApproximator(
                    model=Chain(
                        Dense(ns, 128, relu; init = init),
                        Dense(128, 128, relu; init = init),
                        Dense(128, N * na; init = init),
                    ) |> gpu,
                ),
                stack_size=nothing,
                batch_size=32,
                update_horizon=1,
                min_replay_history=100,
                update_freq=1,
                target_update_freq=100,
                n_quantile=N,
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
ex = E`JuliaRL_QRDQN_CartPole`
run(ex)
plot(ex.hook.rewards)
savefig("assets/JuliaRL_QRDQN_CartPole.png") #hide

# ![](assets/JuliaRL_QRDQN_CartPole.png)
