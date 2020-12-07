function RLCore.Experiment(
    ::Val{:JuliaRL},
    ::Val{:DDPG},
    ::Val{:Pendulum},
    ::Nothing;
    save_dir = nothing,
    seed = 123,
)
    if isnothing(save_dir)
        t = Dates.format(now(), "yyyy_mm_dd_HH_MM_SS")
        save_dir = joinpath(pwd(), "checkpoints", "JuliaRL_DDPG_Pendulum_$(t)")
    end

    lg = TBLogger(joinpath(save_dir, "tb_log"), min_level = Logging.Info)
    rng = StableRNG(seed)
    inner_env = PendulumEnv(T = Float32, rng = rng)
    action_space = get_actions(inner_env)
    low = action_space.low
    high = action_space.high
    ns = length(get_state(inner_env))

    env = inner_env |> ActionTransformedEnv(x -> low + (x + 1) * 0.5 * (high - low))
    init = glorot_uniform(rng)

    create_actor() = Chain(
        Dense(ns, 30, relu; initW = init),
        Dense(30, 30, relu; initW = init),
        Dense(30, 1, tanh; initW = init),
    )

    create_critic() = Chain(
        Dense(ns + 1, 30, relu; initW = init),
        Dense(30, 30, relu; initW = init),
        Dense(30, 1; initW = init),
    )

    agent = Agent(
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
            ρ = 0.995f0,
            batch_size = 64,
            start_steps = 1000,
            start_policy = RandomPolicy(ContinuousSpace(-1.0, 1.0); rng = rng),
            update_after = 1000,
            update_every = 1,
            act_limit = 1.0,
            act_noise = 0.1,
            rng = rng,
        ),
        trajectory = CircularArraySARTTrajectory(
            capacity = 10000,
            state = Vector{Float32} => (ns,),
            action = Float32 => (),
        ),
    )

    stop_condition = StopAfterStep(10_000)
    total_reward_per_episode = TotalRewardPerEpisode()
    time_per_step = TimePerStep()
    hook = ComposedHook(
        total_reward_per_episode,
        time_per_step,
        DoEveryNStep() do t, agent, env
            with_logger(lg) do
                @info(
                    "training",
                    actor_loss = agent.policy.actor_loss,
                    critic_loss = agent.policy.critic_loss
                )
            end
        end,
        DoEveryNEpisode() do t, agent, env
            with_logger(lg) do
                @info "training" reward = total_reward_per_episode.rewards[end] log_step_increment =
                    0
            end
        end,
    )

    Experiment(
        agent,
        env,
        stop_condition,
        hook,
        "# Play Pendulum with DDPG",
    )
end

