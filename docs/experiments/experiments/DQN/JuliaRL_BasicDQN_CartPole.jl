# ---
# title: Play CartPole with BasicDQN
# cover: assets/JuliaRL_BasicDQN_CartPole.png
# description: The simplest example to demonstrate how to use BasicDQN
# date: 2021-05-22
# author: Jun Tian
# ---

#+ tangle=true
using ReinforcementLearning
using StableRNGs
using Flux
using Flux.Losses
using Plots

function RL.Experiment(
    ::Val{:JuliaRL},
    ::Val{:BasicDQN},
    ::Val{:CartPole},
    ::Nothing;
    seed = 123,
)
    rng = StableRNG(seed)
    env = CartPoleEnv(; T = Float32, rng = rng)
    ns, na = length(state(env)), length(action_space(env))

    policy = Agent(
        policy = QBasedPolicy(
            learner = BasicDQNLearner(
                approximator = NeuralNetworkApproximator(
                    model = Chain(
                        Dense(ns, 128, relu; init = glorot_uniform(rng)),
                        Dense(128, 128, relu; init = glorot_uniform(rng)),
                        Dense(128, na; init = glorot_uniform(rng)),
                    ) |> cpu,
                    optimizer = ADAM(),
                ),
                batch_size = 32,
                min_replay_history = 100,
                loss_func = huber_loss,
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
    stop_condition = StopAfterStep(10_000)
    reward_hook = TotalRewardPerEpisode()
    save_fig_hook = DoOnExit() do
        plot(reward_hook.rewards)
        savefig("assets/JuliaRL_BasicDQN_CartPole.png")
    end
    hook = ComposedHook(reward_hook, save_fig_hook)
    Experiment(policy, env, stop_condition, hook, "")
end

#+ tangle=false
ex = E`JuliaRL_BasicDQN_CartPole`;
run(ex)

# The total reward per episode is:
# ![](assets/JuliaRL_BasicDQN_CartPole.png)

# ## References
# ```@docs
# BasicDQNLearner
# ```
