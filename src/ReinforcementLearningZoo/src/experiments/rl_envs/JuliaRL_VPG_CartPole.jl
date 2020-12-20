function RLCore.Experiment(
    ::Val{:JuliaRL},
    ::Val{:VPG},
    ::Val{:CartPole},
    ::Nothing;
    save_dir = nothing,
    seed = 123,
)
    if isnothing(save_dir)
        t = Dates.format(now(), "yyyy_mm_dd_HH_MM_SS")
        save_dir = joinpath(pwd(), "checkpoints", "JuliaRL_VPG_CartPole_$(t)")
    end

    lg = TBLogger(joinpath(save_dir, "tb_log"), min_level = Logging.Info)
    rng = StableRNG(seed)
    env = CartPoleEnv(; T = Float32, rng = rng)
    ns, na = length(state(env)), length(action_space(env))

    agent = Agent(
        policy = VPGPolicy(
            approximator = NeuralNetworkApproximator(
                model = Chain(
                    Dense(ns, 128, relu; initW = glorot_uniform(rng)),
                    Dense(128, 128, relu; initW = glorot_uniform(rng)),
                    Dense(128, na; initW = glorot_uniform(rng)),
                ),
                optimizer = ADAM(),
            ) |> cpu,
            baseline = NeuralNetworkApproximator(
                model = Chain(
                    Dense(ns, 128, relu; initW = glorot_uniform(rng)),
                    Dense(128, 128, relu; initW = glorot_uniform(rng)),
                    Dense(128, 1; initW = glorot_uniform(rng)),
                ),
                optimizer = ADAM(),
            ) |> cpu,
            action_space = action_space(env),
            dist = Categorical,
            γ = 0.99f0,
            rng = rng,
        ),
        trajectory = ElasticSARTTrajectory(
            state = Vector{Float32} => (ns,),
        ),
    )
    # VPG is updated after each episode
    stop_condition = StopAfterEpisode(500)

    total_reward_per_episode = TotalRewardPerEpisode()
    time_per_step = TimePerStep()
    hook = ComposedHook(
        total_reward_per_episode,
        time_per_step,
        DoEveryNEpisode() do t, agent, env
            with_logger(lg) do
                @info(
                    "training",
                    loss = agent.policy.loss,
                    baseline_loss = agent.policy.baseline_loss,
                    reward = total_reward_per_episode.rewards[end],
                )
            end
        end,
    )

    description = "# Play CartPole with VPG"

    Experiment(agent, env, stop_condition, hook, description)
end
