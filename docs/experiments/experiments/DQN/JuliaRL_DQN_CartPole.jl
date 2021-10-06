# ---
# title: JuliaRL\_DQN\_CartPole
# cover: assets/JuliaRL_DQN_CartPole.png
# description: DQN applied to CartPole
# date: 2021-05-22
# author: "[Jun Tian](https://github.com/findmyway)"
# ---

#+ tangle=true
using ReinforcementLearning
using StableRNGs
using Flux
using Flux.Losses

function build_dueling_network(network::Chain)
    lm = length(network)
    if !(network[lm] isa Dense) || !(network[lm-1] isa Dense)
        error("The Qnetwork provided is incompatible with dueling.")
    end
    base = Chain([deepcopy(network[i]) for i in 1:lm-2]...)
    last_layer_dims = size(network[lm].weight, 2)
    val = Chain(deepcopy(network[lm-1]), Dense(last_layer_dims, 1))
    adv = Chain([deepcopy(network[i]) for i in lm-1:lm]...)
    return DuelingNetwork(base, val, adv)
end

function RL.Experiment(
    ::Val{:JuliaRL},
    ::Val{:DQN},
    ::Val{:CartPole},
    ::Nothing;
    seed = 123,
)
    rng = StableRNG(seed)
    env = CartPoleEnv(; T = Float32, rng = rng)
    ns, na = length(state(env)), length(action_space(env))
    base_model = Chain(
        Dense(ns, 128, relu; init = glorot_uniform(rng)),
        Dense(128, 128, relu; init = glorot_uniform(rng)),
        Dense(128, na; init = glorot_uniform(rng)),
    )

    agent = Agent(
        policy = QBasedPolicy(
            learner = DQNLearner(
                approximator = NeuralNetworkApproximator(
                    model = build_dueling_network(base_model) |> cpu,
                    optimizer = ADAM(),
                ),
                target_approximator = NeuralNetworkApproximator(
                    model = build_dueling_network(base_model) |> cpu,
                ),
                loss_func = huber_loss,
                stack_size = nothing,
                batch_size = 32,
                update_horizon = 1,
                min_replay_history = 100,
                update_freq = 1,
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
ex = E`JuliaRL_DQN_CartPole`
run(ex)
plot(ex.hook.rewards)
savefig("assets/JuliaRL_DQN_CartPole.png") #hide

# ![](assets/JuliaRL_DQN_CartPole.png)
