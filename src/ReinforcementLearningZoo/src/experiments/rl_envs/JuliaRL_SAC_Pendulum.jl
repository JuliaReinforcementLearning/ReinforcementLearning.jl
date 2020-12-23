function RLCore.Experiment(
    ::Val{:JuliaRL},
    ::Val{:SAC},
    ::Val{:Pendulum},
    ::Nothing;
    save_dir = nothing,
    seed = 123,
)
    if isnothing(save_dir)
        t = Dates.format(now(), "yyyy_mm_dd_HH_MM_SS")
        save_dir = joinpath(pwd(), "checkpoints", "JuliaRL_SAC_Pendulum_$(t)")
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

    create_policy_net() = NeuralNetworkApproximator(
        model = SACPolicyNetwork(
            Chain(Dense(ns, 30, relu), Dense(30, 30, relu)),
            Chain(Dense(30, 1, initW = init)),
            Chain(Dense(
                30,
                1,
                x -> min(max(x, typeof(x)(-20)), typeof(x)(2)),
                initW = init,
            )),
        ),
        optimizer = ADAM(0.003),
    )

    create_q_net() = NeuralNetworkApproximator(
        model = Chain(
            Dense(ns + 1, 30, relu; initW = init),
            Dense(30, 30, relu; initW = init),
            Dense(30, 1; initW = init),
        ),
        optimizer = ADAM(0.003),
    )

    agent = Agent(
        policy = SACPolicy(
            policy = create_policy_net(),
            qnetwork1 = create_q_net(),
            qnetwork2 = create_q_net(),
            target_qnetwork1 = create_q_net(),
            target_qnetwork2 = create_q_net(),
            γ = 0.99f0,
            ρ = 0.995f0,
            α = 0.2f0,
            batch_size = 64,
            start_steps = 1000,
            start_policy = RandomPolicy(-1.0..1.0; rng = rng),
            update_after = 1000,
            update_every = 1,
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
        DoEveryNEpisode() do t, agent, env
            with_logger(lg) do
                @info "training" reward = total_reward_per_episode.rewards[end]
            end
        end,
    )

    Experiment(agent, env, stop_condition, hook, "# Play Pendulum with SAC")
end
