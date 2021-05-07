function RLCore.Experiment(
    ::Val{:JuliaRL},
    ::Val{:PPO},
    ::Val{:Pendulum},
    ::Nothing;
    save_dir = nothing,
    seed = 123,
)
    if isnothing(save_dir)
        t = Dates.format(now(), "yyyy_mm_dd_HH_MM_SS")
        save_dir = joinpath(pwd(), "checkpoints", "JuliaRL_PPO_Pendulum_$(t)")
    end

    lg = TBLogger(joinpath(save_dir, "tb_log"), min_level = Logging.Info)
    rng = StableRNG(seed)
    inner_env = PendulumEnv(T = Float32, rng = rng)
    A = action_space(inner_env)
    low = A.left
    high = A.right
    ns = length(state(inner_env))

    N_ENV = 8
    UPDATE_FREQ = 2048
    env = MultiThreadEnv([
        PendulumEnv(T = Float32, rng = StableRNG(hash(seed + i))) |>
        env -> ActionTransformedEnv(env, action_mapping = x -> clamp(x * 2, low, high))
        for i in 1:N_ENV
    ])

    init = glorot_uniform(rng)

    agent = Agent(
        policy = PPOPolicy(
            approximator = ActorCritic(
                actor = GaussianNetwork(
                    pre = Chain(
                        Dense(ns, 64, relu; initW = glorot_uniform(rng)),
                        Dense(64, 64, relu; initW = glorot_uniform(rng)),
                    ),
                    μ = Chain(Dense(64, 1, tanh; initW = glorot_uniform(rng)), vec),
                    logσ = Chain(Dense(64, 1; initW = glorot_uniform(rng)), vec),
                ),
                critic = Chain(
                    Dense(ns, 64, relu; initW = glorot_uniform(rng)),
                    Dense(64, 64, relu; initW = glorot_uniform(rng)),
                    Dense(64, 1; initW = glorot_uniform(rng)),
                ),
                optimizer = ADAM(3e-4),
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
            state = Matrix{Float32} => (ns, N_ENV),
            action = Vector{Float32} => (N_ENV,),
            action_log_prob = Vector{Float32} => (N_ENV,),
            reward = Vector{Float32} => (N_ENV,),
            terminal = Vector{Bool} => (N_ENV,),
        ),
    )

    stop_condition = StopAfterStep(50_000)
    total_reward_per_episode = TotalBatchRewardPerEpisode(N_ENV)
    hook = ComposedHook(
        total_reward_per_episode,
        DoEveryNStep() do t, agent, env
            with_logger(lg) do
                @info(
                    "training",
                    actor_loss = agent.policy.actor_loss[end, end],
                    critic_loss = agent.policy.critic_loss[end, end],
                    loss = agent.policy.loss[end, end],
                )
                for i in 1:length(env)
                    if is_terminated(env[i])
                        @info "training" reward = total_reward_per_episode.rewards[i][end] log_step_increment =
                            0
                        break
                    end
                end
            end
        end,
    )

    Experiment(agent, env, stop_condition, hook, "# Play Pendulum with PPO")
end
