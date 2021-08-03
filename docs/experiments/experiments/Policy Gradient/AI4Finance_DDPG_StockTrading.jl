using Statistics
using ReinforcementLearning
using StableRNGs
using Flux
using Distributions

VALIDATION_START_DAY = 1699
REBALANCE_WINDOW = 63
VALIDATION_WINDOW = 63

turbulences = RLEnvs.load_default_stock_data("turbulence.csv")
prices = RLEnvs.load_default_stock_data("prices.csv")
features = RLEnvs.load_default_stock_data("features.csv")

N_DAYS = length(turbulences)

insample_turbulence_threshold = quantile(@view(turbulences[1:VALIDATION_START_DAY-1]), 0.9)
max_turbulence_threshold = quantile(@view(turbulences[1:VALIDATION_START_DAY-1]), 1)


env_train, ppo_env_train, ddpg_policy, ppo_policy = nothing, nothing, nothing, nothing

for validation_start_day in range(VALIDATION_START_DAY, N_DAYS, step=REBALANCE_WINDOW+VALIDATION_START_DAY)
    evaluation_start_day = validation_start_day + VALIDATION_WINDOW
    historical_turbulence_mean = mean(@view turbulences[validation_start_day-VALIDATION_WINDOW:validation_start_day-1])
    if historical_turbulence_mean > insample_turbulence_threshold
        turbulence_threshold = insample_turbulence_threshold
    else
        turbulence_threshold = max_turbulence_threshold
    end

    global env_train
    global ppo_env_train
    global ddpg_policy
    global ppo_policy
    

    env_train = StockTradingEnv(;
        prices=prices,
        features=features,
        first_day=1,
        last_day=validation_start_day-1
    )

    NS = length(state_space(env_train))
    NA = length(action_space(env_train))

    seed = 123

    rng = StableRNG(seed)

    init = glorot_uniform(rng)

    create_actor() = Chain(
        Dense(NS, 64, relu; init = init),
        Dense(64, 64, relu; init = init),
        Dense(64, NA, tanh; init = init),
    )

    create_critic() = Chain(
        Dense(NS + NA, 64, relu; init = init),
        Dense(64, 64, relu; init = init),
        Dense(64, 1; init = init),
    )

    ddpg_policy = Agent(
        policy = DDPGPolicy(
            behavior_actor = NeuralNetworkApproximator(
                model = create_actor(),
                optimizer = ADAM(),
            ),
            behavior_critic = NeuralNetworkApproximator(
                model = create_critic(),
                optimizer = ADAM(),
            ),
            target_actor = NeuralNetworkApproximator(
                model = create_actor(),
                optimizer = ADAM(),
            ),
            target_critic = NeuralNetworkApproximator(
                model = create_critic(),
                optimizer = ADAM(),
            ),
            γ = 0.99f0,
            ρ = 0.999f0,
            na = NA,
            batch_size = 128,
            start_steps = 1000,
            start_policy = RandomPolicy(action_space(env_train); rng = rng),
            update_after = 1000,
            update_every = 1,
            act_limit = 1.0,
            act_noise = 0.1,
            rng = rng,
        ),
        trajectory = CircularArraySARTTrajectory(
            capacity = 50_000,
            state = Vector{Float32} => (NS,),
            action = Float32 => (NA, ),
        ),
    )

    UPDATE_FREQ = 64
    N_ENV = 8

    ppo_env_train = MultiThreadEnv([
        StockTradingEnv(
            prices=prices,
            features=features,
            first_day=1,
            last_day=validation_start_day-1
        ) |>
        env -> ActionTransformedEnv(env, action_mapping = x -> clamp!(x, -1f0, 1f0)) for i in 1:N_ENV
    ])

    ppo_policy = Agent(
        policy = PPOPolicy(
            approximator = ActorCritic(
                actor = GaussianNetwork(
                    pre = Chain(
                        Dense(NS, 64, relu; init = glorot_uniform(rng)),
                        Dense(64, 64, relu; init = glorot_uniform(rng)),
                    ),
                    μ = Dense(64, NA, tanh; init = glorot_uniform(rng)),
                    logσ = Dense(64, NA; init = glorot_uniform(rng)),
                    min_σ=0.01f0,
                    max_σ=100f0,
                ),
                critic = Chain(
                    Dense(NS, 64, relu; init = glorot_uniform(rng)),
                    Dense(64, 64, relu; init = glorot_uniform(rng)),
                    Dense(64, 1; init = glorot_uniform(rng)),
                ),
                optimizer = ADAM(),
            ) |> cpu,
            γ = 0.99f0,
            λ = 0.95f0,
            clip_range = 0.2f0,
            max_grad_norm = 0.5f0,
            n_epochs = 10,
            n_microbatches = 32,
            actor_loss_weight = 1.0f0,
            critic_loss_weight = 0.5f0,
            entropy_loss_weight = 0.00f0,
            dist = Normal,
            rng = rng,
            update_freq = UPDATE_FREQ,
        ),
        trajectory = PPOTrajectory(;
            capacity = UPDATE_FREQ,
            state = Matrix{Float32} => (NS, N_ENV),
            action = Vector{Float32} => (NA, N_ENV,),
            action_log_prob = Vector{Float32} => (N_ENV,),
            reward = Vector{Float32} => (N_ENV,),
            terminal = Vector{Bool} => (N_ENV,),
        ),
    )
    break
end