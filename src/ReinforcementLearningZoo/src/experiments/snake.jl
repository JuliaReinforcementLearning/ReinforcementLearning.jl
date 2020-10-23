function RLCore.Experiment(
    ::Val{:JuliaRL},
    ::Val{:DQN},
    ::Val{:SnakeGame},
    ::Nothing;
    save_dir = nothing,
    seed = 123,
)
    rng = StableRNG(seed)

    SHAPE = (8, 8)
    inner_env = SnakeGameEnv(; action_style = FULL_ACTION_SET, shape = SHAPE, rng = rng)

    board_size = size(get_state(inner_env))
    N_FRAMES = 4

    env =
        inner_env |>
        StateOverriddenEnv(StackFrames(board_size..., N_FRAMES),) |>
        StateCachedEnv

    N_ACTIONS = length(get_actions(env))

    if isnothing(save_dir)
        t = Dates.format(now(), "yyyy_mm_dd_HH_MM_SS")
        save_dir = joinpath(pwd(), "checkpoints", "SnakeGame_$(t)")
    end

    lg = TBLogger(joinpath(save_dir, "tb_log"), min_level = Logging.Info)

    init = glorot_uniform(rng)

    update_freq = 4

    create_model() =
        Chain(
            x -> reshape(x, SHAPE..., :, size(x, ndims(x))),
            CrossCor(
                (3, 3),
                board_size[end] * N_FRAMES => 16,
                relu;
                stride = 1,
                pad = 1,
                init = init,
            ),
            CrossCor((3, 3), 16 => 32, relu; stride = 1, pad = 1, init = init),
            x -> reshape(x, :, size(x, ndims(x))),
            Dense(8 * 8 * 32, 256, relu; initW = init),
            Dense(256, N_ACTIONS; initW = init),
        ) |> cpu

    agent = Agent(
        policy = QBasedPolicy(
            learner = DQNLearner(
                approximator = NeuralNetworkApproximator(
                    model = create_model(),
                    optimizer = ADAM(0.001),
                ),
                target_approximator = NeuralNetworkApproximator(model = create_model()),
                update_freq = update_freq,
                γ = 0.99f0,
                update_horizon = 1,
                batch_size = 32,
                stack_size = nothing,
                min_replay_history = 20_000,
                loss_func = huber_loss,
                target_update_freq = 8_000,
                rng = rng,
            ),
            explorer = EpsilonGreedyExplorer(
                ϵ_init = 1.0,
                ϵ_stable = 0.01,
                decay_steps = 250_000,
                kind = :linear,
                rng = rng,
            ),
        ),
        trajectory = CircularCompactSALRTSALTrajectory(
            capacity = 500_000,
            state_type = Float32,
            state_size = (board_size..., N_FRAMES),
            legal_actions_mask_size = (N_ACTIONS,),
        ),
    )

    evaluation_result = []
    EVALUATION_FREQ = 100_000
    N_CHECKPOINTS = 3

    total_reward_per_episode = TotalRewardPerEpisode()
    time_per_step = TimePerStep()
    steps_per_episode = StepsPerEpisode()
    hook = ComposedHook(
        total_reward_per_episode,
        time_per_step,
        steps_per_episode,
        DoEveryNStep(update_freq) do t, agent, env
            with_logger(lg) do
                @info "training" loss = agent.policy.learner.loss log_step_increment =
                    update_freq
            end
        end,
        DoEveryNEpisode() do t, agent, env
            with_logger(lg) do
                @info "training" episode_length = steps_per_episode.steps[end] reward =
                    total_reward_per_episode.rewards[end] log_step_increment = 0
            end
        end,
    )

    N_TRAINING_STEPS = 1_000_000
    stop_condition = StopAfterStep(N_TRAINING_STEPS)
    description = """
    # Play Single Agent SnakeGame with DQN

    You can view the tensorboard logs with `tensorboard --logdir $(joinpath(save_dir, "tb_log"))`
    """
    Experiment(agent, env, stop_condition, hook, description)
end
