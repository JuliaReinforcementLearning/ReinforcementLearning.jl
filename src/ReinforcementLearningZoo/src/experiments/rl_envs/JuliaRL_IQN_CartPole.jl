function RLCore.Experiment(
    ::Val{:JuliaRL},
    ::Val{:IQN},
    ::Val{:CartPole},
    ::Nothing;
    save_dir = nothing,
    seed = 123,
)
    if isnothing(save_dir)
        t = Dates.format(now(), "yyyy_mm_dd_HH_MM_SS")
        save_dir = joinpath(pwd(), "checkpoints", "JuliaRL_IQN_CartPole_$(t)")
    end

    lg = TBLogger(joinpath(save_dir, "tb_log"), min_level = Logging.Info)
    rng = StableRNG(seed)

    env = CartPoleEnv(; T = Float32, rng = rng)
    ns, na = length(state(env)), length(action_space(env))

    init = glorot_uniform(rng)
    Nₑₘ = 16
    n_hidden = 64
    κ = 1.0f0

    nn_creator() =
        ImplicitQuantileNet(
            ψ = Dense(ns, n_hidden, relu; initW = init),
            ϕ = Dense(Nₑₘ, n_hidden, relu; initW = init),
            header = Dense(n_hidden, na; initW = init),
        ) |> cpu

    agent = Agent(
        policy = QBasedPolicy(
            learner = IQNLearner(
                approximator = NeuralNetworkApproximator(
                    model = nn_creator(),
                    optimizer = ADAM(0.001),
                ),
                target_approximator = NeuralNetworkApproximator(model = nn_creator()),
                κ = κ,
                N = 8,
                N′ = 8,
                Nₑₘ = Nₑₘ,
                K = 32,
                γ = 0.99f0,
                stack_size = nothing,
                batch_size = 32,
                update_horizon = 1,
                min_replay_history = 100,
                update_freq = 1,
                target_update_freq = 100,
                default_priority = 1.0f2,
                rng = rng,
                device_rng = rng,
            ),
            explorer = EpsilonGreedyExplorer(
                kind = :exp,
                ϵ_stable = 0.01,
                decay_steps = 500,
                rng = rng,
            ),
        ),
        trajectory = CircularArrayPSARTTrajectory(
            capacity = 1000,
            state = Vector{Float32} => (ns,),
        ),
    )

    stop_condition = StopAfterStep(10_000)

    total_reward_per_episode = TotalRewardPerEpisode()
    time_per_step = TimePerStep()
    hook = ComposedHook(
        total_reward_per_episode,
        time_per_step,
        DoEveryNStep() do t, agent, env
            if agent.policy.learner.update_step % agent.policy.learner.update_freq == 0
                with_logger(lg) do
                    @info "training" loss = agent.policy.learner.loss
                end
            end
        end,
        DoEveryNEpisode() do t, agent, env
            with_logger(lg) do
                @info "training" reward = total_reward_per_episode.rewards[end] log_step_increment =
                    0
            end
        end,
    )

    description = """
    This experiment uses the `IQNLearner` method with a `ImplicitQuantileNet`.
    The testing environment is CartPoleEnv.
    """

    Experiment(agent, env, stop_condition, hook, description)
end
