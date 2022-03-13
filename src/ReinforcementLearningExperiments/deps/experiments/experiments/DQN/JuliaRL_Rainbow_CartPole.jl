# ---
# title: JuliaRL\_Rainbow\_CartPole
# cover: assets/JuliaRL_Rainbow_CartPole.png
# description: Rainbow applied to CartPole
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
    ::Val{:Rainbow},
    ::Val{:CartPole},
    ::Nothing;
    seed = 123,
)
    rng = StableRNG(seed)

    env = CartPoleEnv(; T = Float32, rng = rng)
    ns, na = length(state(env)), length(action_space(env))

    n_atoms = 51
    agent = Agent(
        policy = QBasedPolicy(
            learner = RainbowLearner(
                approximator = NeuralNetworkApproximator(
                    model = Chain(
                        Dense(ns, 128, relu; init = glorot_uniform(rng)),
                        Dense(128, 128, relu; init = glorot_uniform(rng)),
                        Dense(128, na * n_atoms; init = glorot_uniform(rng)),
                    ) |> gpu,
                    optimizer = ADAM(0.0005),
                ),
                target_approximator = NeuralNetworkApproximator(
                    model = Chain(
                        Dense(ns, 128, relu; init = glorot_uniform(rng)),
                        Dense(128, 128, relu; init = glorot_uniform(rng)),
                        Dense(128, na * n_atoms; init = glorot_uniform(rng)),
                    ) |> gpu,
                    optimizer = ADAM(0.0005),
                ),
                n_actions = na,
                n_atoms = n_atoms,
                Vₘₐₓ = 200.0f0,
                Vₘᵢₙ = 0.0f0,
                update_freq = 1,
                γ = 0.99f0,
                update_horizon = 1,
                batch_size = 32,
                stack_size = nothing,
                min_replay_history = 100,
                loss_func = (ŷ, y) -> logitcrossentropy(ŷ, y; agg = identity),
                target_update_freq = 100,
                rng = rng,
            ),
            explorer = EpsilonGreedyExplorer(
                kind = :exp,
                ϵ_stable = 0.01,
                decay_steps = 500,
                rng = rng,
            ),
        ),
        trajectory = CircularArrayPSARTTrajectory(
            capacity = 1000,
            state = Vector{Float32} => (ns,),
        ),
    )

    stop_condition = StopAfterStep(10_000, is_show_progress=!haskey(ENV, "CI"))
    hook = TotalRewardPerEpisode()
    Experiment(agent, env, stop_condition, hook, "")
end

#+ tangle=false
using Plots
pyplot() #hide
ex = E`JuliaRL_Rainbow_CartPole`
run(ex)
plot(ex.hook.rewards)
savefig("assets/JuliaRL_Rainbow_CartPole.png") #hide

# ![](assets/JuliaRL_Rainbow_CartPole.png)
