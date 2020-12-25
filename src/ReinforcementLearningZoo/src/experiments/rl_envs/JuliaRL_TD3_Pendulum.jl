function RLCore.Experiment(
    ::Val{:JuliaRL},
    ::Val{:TD3},
    ::Val{:Pendulum},
    ::Nothing;
    save_dir = nothing,
    seed = 123,
)
    if isnothing(save_dir)
        t = Dates.format(now(), "yyyy_mm_dd_HH_MM_SS")
        save_dir = joinpath(pwd(), "checkpoints", "JuliaRL_TD3_Pendulum_$(t)")
    end

    lg = TBLogger(joinpath(save_dir, "tb_log"), min_level = Logging.Info)
    rng = StableRNG(seed)
    inner_env = PendulumEnv(T = Float32, rng = rng)
    A = action_space(inner_env)
    low = A.left
    high = A.right
    ns = length(state(inner_env))

    env = ActionTransformedEnv(
        inner_env;
        action_mapping = x -> low + (x + 1) * 0.5 * (high - low),
    )
    init = glorot_uniform(rng)

    create_actor() = Chain(
        Dense(ns, 30, relu; initW = init),
        Dense(30, 30, relu; initW = init),
        Dense(30, 1, tanh; initW = init),
    )

    create_critic_model() = Chain(
        Dense(ns + 1, 30, relu; initW = init),
        Dense(30, 30, relu; initW = init),
        Dense(30, 1; initW = init),
    )

    create_critic() = TD3Critic(create_critic_model(), create_critic_model())

    agent = Agent(
        policy = TD3Policy(
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
            ρ = 0.99f0,
            batch_size = 64,
            start_steps = 1000,
            start_policy = RandomPolicy(-1.0..1.0; rng = rng),
            update_after = 1000,
            update_every = 1,
            policy_freq = 2,
            target_act_limit = 1.0,
            target_act_noise = 0.1,
            act_limit = 1.0,
            act_noise = 0.1,
            rng = rng,
        ),
        trajectory = CircularArraySARTTrajectory(
            capacity = 10_000,
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

    Experiment(agent, env, stop_condition, hook, "# Play Pendulum with TD3")
end
