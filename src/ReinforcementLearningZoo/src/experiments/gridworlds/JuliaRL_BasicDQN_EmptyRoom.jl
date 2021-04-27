function RLCore.Experiment(
    ::Val{:JuliaRL},
    ::Val{:BasicDQN},
    ::Val{:EmptyRoom},
    ::Nothing;
    seed = 123,
    save_dir = nothing,
)
    if isnothing(save_dir)
        t = Dates.format(now(), "yyyy_mm_dd_HH_MM_SS")
        save_dir = joinpath(pwd(), "checkpoints", "JuliaRL_BasicDQN_EmptyRoom$(t)")
    end
    log_dir = joinpath(save_dir, "tb_log")
    lg = TBLogger(log_dir, min_level = Logging.Info)
    rng = StableRNG(seed)

    inner_env = GridWorlds.EmptyRoomDirected(rng = rng)
    action_space_mapping = x -> Base.OneTo(length(RLBase.action_space(inner_env)))
    action_mapping = i -> RLBase.action_space(inner_env)[i]
    env = RLEnvs.ActionTransformedEnv(
        inner_env,
        action_space_mapping = action_space_mapping,
        action_mapping = action_mapping,
    )
    env = RLEnvs.StateOverriddenEnv(env, x -> vec(Float32.(x)))
    env = RewardOverriddenEnv(env, x -> x - convert(typeof(x), 0.01))
    env = MaxTimeoutEnv(env, 240)

    ns, na = length(state(env)), length(action_space(env))
    agent = Agent(
        policy = QBasedPolicy(
            learner = BasicDQNLearner(
                approximator = NeuralNetworkApproximator(
                    model = Chain(
                        Dense(ns, 128, relu; initW = glorot_uniform(rng)),
                        Dense(128, 128, relu; initW = glorot_uniform(rng)),
                        Dense(128, na; initW = glorot_uniform(rng)),
                    ) |> cpu,
                    optimizer = ADAM(),
                ),
                batch_size = 32,
                min_replay_history = 100,
                loss_func = huber_loss,
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
            with_logger(lg) do
                @info "training" loss = agent.policy.learner.loss
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
    This experiment uses three dense layers to approximate the Q value.
    The testing environment is EmptyRoom.

    You can view the runtime logs with `tensorboard --logdir $log_dir`.
    Some useful statistics are stored in the `hook` field of this experiment.
    """

    Experiment(agent, env, stop_condition, hook, description)
end
