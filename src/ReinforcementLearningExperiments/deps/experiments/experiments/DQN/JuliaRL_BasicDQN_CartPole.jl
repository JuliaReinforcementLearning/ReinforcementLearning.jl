# ---
# title: Generate your cover on the fly
# cover: assets/logo.svg
# description: this demo shows you how to generate cover on the fly
# date: 2020-09-13
# author: Jun Tian
# ---

# There're many reasons that you don't want to mannually manage the cover image.
# DemoCards.jl allows you to generate the card cover on the fly for demos written
# in julia.
#
# Let's do this with a simple example

#+ tangle=true
using ReinforcementLearning
using StableRNGs
using Flux
using Flux.Losses

function RL.Experiment(
    ::Val{:JuliaRL},
    ::Val{:BasicDQN},
    ::Val{:CartPole},
    ::Nothing;
    seed = 123,
    save_dir = nothing,
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
    hook = ComposedHook(TotalRewardPerEpisode(), TimePerStep())
    Experiment(policy, env, stop_condition, hook, "")
end

# Now let's get start!

#+ tangle=false
run(RL.Experiment(Val(:JuliaRL), Val(:BasicDQN), Val(:CartPole), nothing))

1 + 2