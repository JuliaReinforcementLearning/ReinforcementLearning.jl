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
    na = 1

    env = ActionTransformedEnv(
        inner_env;
        action_mapping = x -> low + (x[1] + 1) * 0.5 * (high - low),
    )
    init = glorot_uniform(rng)

    create_policy_net() = NeuralNetworkApproximator(
        model = GaussianNetwork(
            pre = Chain(Dense(ns, 30, relu), Dense(30, 30, relu)),
            μ = Chain(Dense(30, na, initW = init)),
            logσ = Chain(
                Dense(30, na, x -> clamp.(x, typeof(x)(-10), typeof(x)(2)), initW = init),
            ),
        ),
        optimizer = ADAM(0.003),
    )

    create_q_net() = NeuralNetworkApproximator(
        model = Chain(
            Dense(ns + na, 30, relu; initW = init),
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
            start_policy = RandomPolicy(Space([-1.0..1.0 for _ in 1:na]); rng = rng),
            update_after = 1000,
            update_every = 1,
            rng = rng,
        ),
        trajectory = CircularArraySARTTrajectory(
            capacity = 10000,
            state = Vector{Float32} => (ns,),
            action = Vector{Float32} => (na,),
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
                    reward_term = agent.policy.reward_term,
                    entropy_term = agent.policy.entropy_term,
                )
                if is_terminated(env)
                    @info "training" reward = total_reward_per_episode.reward log_step_increment =
                        0
                end
            end
        end,
    )

    Experiment(agent, env, stop_condition, hook, "# Play Pendulum with SAC")
end
