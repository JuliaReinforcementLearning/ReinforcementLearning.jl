function RLCore.Experiment(
    ::Val{:JuliaRL},
    ::Val{:DQN},
    ::Val{:CartPole},
    ::Nothing;
    save_dir = nothing,
    seed = 123,
)
    if isnothing(save_dir)
        t = Dates.format(now(), "yyyy_mm_dd_HH_MM_SS")
        save_dir = joinpath(pwd(), "checkpoints", "JuliaRL_DQN_CartPole_$(t)")
    end

    lg = TBLogger(joinpath(save_dir, "tb_log"), min_level = Logging.Info)
    rng = StableRNG(seed)

    env = CartPoleEnv(; T = Float32, rng = rng)
    ns, na = length(state(env)), length(action_space(env))
    base_model = Chain(
        Dense(ns, 128, relu; initW = glorot_uniform(rng)),
        Dense(128, 128, relu; initW = glorot_uniform(rng)),
        Dense(128, na; initW = glorot_uniform(rng)),
    )

    agent = Agent(
        policy = QBasedPolicy(
            learner = DQNLearner(
                approximator = NeuralNetworkApproximator(
                    model = build_dueling_network(base_model) |> cpu,
                    optimizer = ADAM(),
                ),
                target_approximator = NeuralNetworkApproximator(
                    model = build_dueling_network(base_model) |> cpu,
                ),
                loss_func = huber_loss,
                stack_size = nothing,
                batch_size = 32,
                update_horizon = 1,
                min_replay_history = 100,
                update_freq = 1,
                target_update_freq = 100,
                rng = rng,
            ),
            explorer = EpsilonGreedyExplorer(
                kind = :exp,
                ϵ_stable = 0.01,
                decay_steps = 500,
                rng = rng,
            ),
        ),
        trajectory = CircularArraySARTTrajectory(
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
    This experiment uses the `DQNLearner` method with three dense layers to approximate the Q value.
    The testing environment is CartPoleEnv.
    """

    Experiment(agent, env, stop_condition, hook, description)
end
