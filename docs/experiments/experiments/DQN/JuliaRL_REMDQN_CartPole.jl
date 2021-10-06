# ---
# title: JuliaRL\_REMDQN\_CartPole
# cover: assets/JuliaRL_REMDQN_CartPole.png
# description: REMDQN applied to CartPole
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
    ::Val{:REMDQN},
    ::Val{:CartPole},
    ::Nothing;
    seed = 123,
)
    rng = StableRNG(seed)

    env = CartPoleEnv(; T = Float32, rng = rng)
    ns, na = length(state(env)), length(action_space(env))
    ensemble_num = 16

    agent = Agent(
        policy = QBasedPolicy(
            learner = REMDQNLearner(
                approximator = NeuralNetworkApproximator(
                    model = Chain(
                        ## Multi-head method, please refer to "https://github.com/google-research/batch_rl/tree/b55ba35ebd2381199125dd77bfac9e9c59a64d74/batch_rl/multi_head".
                        Dense(ns, 128, relu; init = glorot_uniform(rng)),
                        Dense(128, 128, relu; init = glorot_uniform(rng)),
                        Dense(128, na * ensemble_num; init = glorot_uniform(rng)),
                    ) |> cpu,
                    optimizer = ADAM(),
                ),
                target_approximator = NeuralNetworkApproximator(
                    model = Chain(
                        Dense(ns, 128, relu; init = glorot_uniform(rng)),
                        Dense(128, 128, relu; init = glorot_uniform(rng)),
                        Dense(128, na * ensemble_num; init = glorot_uniform(rng)),
                    ) |> cpu,
                ),
                loss_func = huber_loss,
                stack_size = nothing,
                batch_size = 32,
                update_horizon = 1,
                min_replay_history = 100,
                update_freq = 1,
                target_update_freq = 100,
                ensemble_num = ensemble_num,
                ensemble_method = :rand,
                rng = rng,
            ),
            explorer = EpsilonGreedyExplorer(
                kind = :exp,
                ϵ_stable = 0.01,
                decay_steps = 500,
                rng = rng,
            ),
        ),
        trajectory = CircularArraySARTTrajectory(
            capacity = 1000,
            state = Vector{Float32} => (ns,),
        ),
    )

    stop_condition = StopAfterStep(10_000, is_show_progress = !haskey(ENV, "CI"))
    hook = TotalRewardPerEpisode()
    Experiment(agent, env, stop_condition, hook, "")
end

#+ tangle=false
using Plots
pyplot() #hide
ex = E`JuliaRL_REMDQN_CartPole`
run(ex)
plot(ex.hook.rewards)
savefig("assets/JuliaRL_REMDQN_CartPole.png") #hide

# ![](assets/JuliaRL_REMDQN_CartPole.png)
