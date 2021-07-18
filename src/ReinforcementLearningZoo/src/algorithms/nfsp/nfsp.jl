export NFSPAgent

using Distributions: TruncatedNormal

"""
    NFSPAgent(; η, rng, rl_agent::Agent, sl_agent::Agent)

Neural Fictitious Self-Play (NFSP) agent implemented in Julia. 
See the paper https://arxiv.org/abs/1603.01121 for more details.

# Keyword arguments

- `η`, anticipatory parameter, the probability to use `ϵ-greedy(Q)` policy when training the agent.
- `rng=StableRNG(seed)`.
- `rl_agent::Agent`, Reinforcement Learning(RL) agent(use `DQN` as follows), which works to search the best response from the self-play process.
- `sl_agent::Agent`, Supervisor Learning(SL) agent(use `AverageLearner` as follows), which works to learn the best response from the rl_agent's policy.
"""
mutable struct NFSPAgent <: AbstractPolicy
    η
    rng
    rl_agent::Agent
    sl_agent::Agent
end

# parameters initial method for network
function _TruncatedNormal(out_size, in_size)
    mean, stddev = 0.0, 1.0 / sqrt(in_size)
    lower, upper = (-2 * stddev - mean) / stddev, (2 * stddev - mean) / stddev
    d = TruncatedNormal(mean, stddev, lower, upper)
    rand(d, (out_size, in_size))
end


# DQNLearner relative function
function build_dueling_network(network::Chain)
    lm = length(network)
    if !(network[lm] isa Dense) || !(network[lm-1] isa Dense)
        error("The Qnetwork provided is incompatible with dueling.")
    end
    base = Chain([deepcopy(network[i]) for i=1:lm-2]...)
    last_layer_dims = size(network[lm].W, 2)
    val = Chain(deepcopy(network[lm-1]), Dense(last_layer_dims, 1))
    adv = Chain([deepcopy(network[i]) for i=lm-1:lm]...)
    return DuelingNetwork(base, val, adv)
end


function NFSPAgent(
    env::AbstractEnv,
    player;
    # parameters setting
    # public parameters
    η = 0.1f0,
    _device = Flux.cpu,
    Optimizer = Flux.Descent,
    rng = StableRNG(123),
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
    base_model = Chain(
        Dense(ns, hidden_layers[1], relu; init = _TruncatedNormal),
        [Dense(hidden_layers[i], hidden_layers[i+1], relu; init = _TruncatedNormal) 
            for i in 1:length(hidden_layers)-1]...,
        Dense(hidden_layers[end], na, relu; init = _TruncatedNormal)
    )

    # RL agent
    rl_agent = Agent(
        policy = QBasedPolicy(
            learner = DQNLearner(
                approximator = NeuralNetworkApproximator(
                    model = build_dueling_network(base_model) |> _device,
                    optimizer = Optimizer(rl_learning_rate),
                ),
                target_approximator = NeuralNetworkApproximator(
                    model = build_dueling_network(base_model) |> _device,
                ),
                γ = discount_factor,
                loss_func = rl_loss_func,
                batch_size = batch_size,
                update_freq = learn_freq,
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
            state = Vector{Int64} => (ns, )
        ),
    )

    # SL agent
    sl_agent = Agent(
        policy = QBasedPolicy(
            learner = AverageLearner(
                approximator = NeuralNetworkApproximator(
                    model = base_model |> _device,
                    optimizer = Optimizer(sl_learning_rate),
                ),
                batch_size = batch_size,
                update_freq = learn_freq,
                min_reservoir_history = min_buffer_size_to_learn,
                rng = rng,
            ),
            explorer = WeightedSoftmaxExplorer(),
        ),
        trajectory = CircularArraySARTTrajectory(
            capacity = reservoir_buffer_capacity,
            state = Vector{Int64} => (ns, )
        ),
    )

    NFSPAgent(
        η,
        rng,
        rl_agent,
        sl_agent,
    )
end


(π::NFSPAgent)(env::AbstractEnv) = π.sl_agent(env)


RLBase.prob(π::NFSPAgent, env::AbstractEnv, args...) = prob(π.sl_agent.policy, env, args...)


function RLBase.update!(π::NFSPAgent, env::AbstractEnv)
    sl = π.sl_agent
    rl = π.rl_agent
    
    if rand(π.rng) < π.η
        action = rl(env)
        sl(PRE_ACT_STAGE, env, action)
        rl(PRE_ACT_STAGE, env, action)
        env(action)
        sl(POST_ACT_STAGE, env)
        rl(POST_ACT_STAGE, env)
    else
        action = sl(env)
        rl(PRE_ACT_STAGE, env, action)
        env(action)
        rl(POST_ACT_STAGE, env)
    end
end