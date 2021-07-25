export NFSPAgent

using Distributions: TruncatedNormal

"""
    NFSPAgent(; η, rng, rl_agent::Agent, sl_agent::Agent)

Neural Fictitious Self-Play (NFSP) agent implemented in Julia. 
See the paper https://arxiv.org/abs/1603.01121 for more details.

# Keyword arguments

- `rl_agent::Agent`, Reinforcement Learning(RL) agent(use `DQN` as follows), which works to search the best response from the self-play process.
- `sl_agent::Agent`, Supervisor Learning(SL) agent(use `BehaviorCloningPolicy` as follows), which works to learn the best response from the rl_agent's policy.
- `η`, anticipatory parameter, the probability to use `ϵ-greedy(Q)` policy when training the agent.
- `rng=Random.GLOBAL_RNG`.
- `update_freq::Int`: the frequency of updating the agents' `approximator`.
- `step_counter::Int`, count the step.
"""
mutable struct NFSPAgent <: AbstractPolicy
    rl_agent::Agent
    sl_agent::Agent
    η
    rng
    update_freq::Int
    step_counter::Int
    mode::Bool
end

function NFSPAgent(
    env::AbstractEnv,
    player;
    # parameters setting
    # public parameters
    η = 0.1f0,
    _device = Flux.cpu,
    Optimizer = Flux.Descent,
    rng = Random.GLOBAL_RNG,
    batch_size::Int = 128,
    learn_freq::Int = 128,
    min_buffer_size_to_learn::Int = 1000,
    hidden_layers = (128, 128),

    # Reinforcement Learning(RL) agent parameters
    rl_loss_func = mse,
    rl_learning_rate = 0.01,
    replay_buffer_capacity::Int = 200_000,
    ϵ_start = 0.06,
    ϵ_end = 0.001,
    ϵ_decay = 20_000_000,
    discount_factor::Float32 = 1.0f0,
    update_target_network_freq::Int = 19200,

    # Supervisor Learning(SL) agent parameters
    sl_learning_rate = 0.01,
    reservoir_buffer_capacity::Int = 2_000_000
    )

    # base Neural network for training
    ns, na = length(state(env, player)), length(action_space(env, player))

    # RL agent
    rl_agent = Agent(
        policy = QBasedPolicy(
            learner = DQNLearner(
                approximator = NeuralNetworkApproximator(
                    model = Chain(
                        Dense(ns, hidden_layers[1], relu; init = glorot_normal(rng)),
                        [Dense(hidden_layers[i], hidden_layers[i+1], relu; init = glorot_normal(rng)) 
                            for i in 1:length(hidden_layers)-1]...,
                        Dense(hidden_layers[end], na; init = glorot_normal(rng))
                    ) |> _device,
                    optimizer = Optimizer(rl_learning_rate),
                ),
                target_approximator = NeuralNetworkApproximator(
                    model = Chain(
                        Dense(ns, hidden_layers[1], relu; init = glorot_normal(rng)),
                        [Dense(hidden_layers[i], hidden_layers[i+1], relu; init = glorot_normal(rng)) 
                            for i in 1:length(hidden_layers)-1]...,
                        Dense(hidden_layers[end], na; init = glorot_normal(rng))
                    ) |> _device,
                ),
                γ = discount_factor,
                loss_func = rl_loss_func,
                batch_size = batch_size,
                update_freq = learn_freq,
                update_horizon = 0,
                min_replay_history = min_buffer_size_to_learn,
                target_update_freq = update_target_network_freq,
                rng = rng,
            ),
            explorer = EpsilonGreedyExplorer(
                kind = :linear,
                ϵ_init = ϵ_start,
                ϵ_stable = ϵ_end,
                decay_steps = ϵ_decay,
                rng = rng,
            ),
        ),
        trajectory = CircularArraySARTTrajectory(
            capacity = replay_buffer_capacity,
            state = Vector{Float64} => (ns, )
        ),
    )

    # SL agent
    sl_agent = Agent(
        policy = BehaviorCloningPolicy(;
            approximator = NeuralNetworkApproximator(
                model = Chain(
                    Dense(ns, hidden_layers[1], relu; init = glorot_normal(rng)),
                    [Dense(hidden_layers[i], hidden_layers[i+1], relu; init = glorot_normal(rng)) 
                        for i in 1:length(hidden_layers)-1]...,
                    Dense(hidden_layers[end], na; init = glorot_normal(rng))
                ) |> _device,
                optimizer = Optimizer(sl_learning_rate),
            ),
            explorer = WeightedSoftmaxExplorer(),
            batch_size = batch_size,
            min_reservoir_history = min_buffer_size_to_learn,
            rng = rng,
        ),
        trajectory = ReservoirTrajectory(
            reservoir_buffer_capacity;
            rng = rng,
            :state => Vector{Float64},
            :action_probs => Vector{Float64},
        ),
    )

    NFSPAgent(
        rl_agent,
        sl_agent,
        η,
        rng,
        learn_freq,
        0,
        true,
    )
end
