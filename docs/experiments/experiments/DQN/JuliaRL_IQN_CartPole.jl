# ---
# title: JuliaRL\_IQN\_CartPole
# cover: assets/JuliaRL_IQN_CartPole.png
# description: IQN applied to CartPole
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
    ::Val{:IQN},
    ::Val{:CartPole},
    ::Nothing;
    seed = 123,
)
    rng = StableRNG(seed)
    env = CartPoleEnv(; T = Float32, rng = rng)
    ns, na = length(state(env)), length(action_space(env))
    init = glorot_uniform(rng)
    Nₑₘ = 16
    n_hidden = 64
    κ = 1.0f0

    nn_creator() =
        ImplicitQuantileNet(
            ψ = Dense(ns, n_hidden, relu; init = init),
            ϕ = Dense(Nₑₘ, n_hidden, relu; init = init),
            header = Dense(n_hidden, na; init = init),
        ) |> cpu

    agent = Agent(
        policy = QBasedPolicy(
            learner = IQNLearner(
                approximator = NeuralNetworkApproximator(
                    model = nn_creator(),
                    optimizer = ADAM(0.001),
                ),
                target_approximator = NeuralNetworkApproximator(model = nn_creator()),
                κ = κ,
                N = 8,
                N′ = 8,
                Nₑₘ = Nₑₘ,
                K = 32,
                γ = 0.99f0,
                stack_size = nothing,
                batch_size = 32,
                update_horizon = 1,
                min_replay_history = 100,
                update_freq = 1,
                target_update_freq = 100,
                default_priority = 1.0f2,
                rng = rng,
                device_rng = rng,
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

    stop_condition = StopAfterStep(10_000, is_show_progress = !haskey(ENV, "CI"))
    hook = TotalRewardPerEpisode()
    Experiment(agent, env, stop_condition, hook, "")
end

#+ tangle=false
using Plots
pyplot() #hide
ex = E`JuliaRL_IQN_CartPole`
run(ex)
plot(ex.hook.rewards)
savefig("assets/JuliaRL_IQN_CartPole.png") #hide

# ![](assets/JuliaRL_IQN_CartPole.png)
