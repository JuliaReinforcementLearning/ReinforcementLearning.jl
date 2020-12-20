function RLCore.Experiment(
    ::Val{:JuliaRL},
    ::Val{:BasicDQN},
    ::Val{:MountainCar},
    ::Nothing;
    save_dir = nothing,
    seed = 123,
)
    rng = StableRNG(seed)
    if isnothing(save_dir)
        t = Dates.format(now(), "yyyy_mm_dd_HH_MM_SS")
        save_dir = joinpath(pwd(), "checkpoints", "JuliaRL_BasicDQN_MountainCar_$(t)")
    end

    lg = TBLogger(joinpath(save_dir, "tb_log"), min_level = Logging.Info)

    env = MountainCarEnv(; T = Float32, max_steps = 5000, rng = rng)
    ns, na = length(state(env)), length(action_space(env))
    agent = Agent(
        policy = QBasedPolicy(
            learner = BasicDQNLearner(
                approximator = NeuralNetworkApproximator(
                    model = Chain(
                        Dense(ns, 64, relu; initW = glorot_uniform(rng)),
                        Dense(64, 64, relu; initW = glorot_uniform(rng)),
                        Dense(64, na; initW = glorot_uniform(rng)),
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
            capacity = 50_000,
            state = Vector{Float32} => (ns,),
        ),
    )

    stop_condition = StopAfterStep(70_000)

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
    The testing environment is MountainCarEnv.
    """

    Experiment(agent, env, stop_condition, hook, description)
end
