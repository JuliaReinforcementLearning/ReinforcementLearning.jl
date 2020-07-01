export Experiment

using Dates
using ReinforcementLearningBase
using ReinforcementLearningCore
using .ReinforcementLearningEnvironments
using Flux
using BSON
using TensorBoardLogger
using Logging

function RLCore.Experiment(
    ::Val{:JuliaRL},
    ::Val{:BasicDQN},
    ::Val{:CartPole},
    ::Nothing;
    save_dir = nothing,
)
    if isnothing(save_dir)
        t = Dates.format(now(), "yyyymmddHHMMSS")
        save_dir = joinpath(pwd(), "checkpoints", "JuliaRL_BasicDQN_CartPole_$(t)")
    end

    lg = TBLogger(joinpath(save_dir, "tb_log"), min_level = Logging.Info)

    env = CartPoleEnv(; T = Float32, seed = 11)
    ns, na = length(rand(get_observation_space(env))), length(get_action_space(env))
    agent = Agent(
        policy = QBasedPolicy(
            learner = BasicDQNLearner(
                approximator = NeuralNetworkApproximator(
                    model = Chain(
                        Dense(ns, 128, relu; initW = seed_glorot_uniform(seed = 17)),
                        Dense(128, 128, relu; initW = seed_glorot_uniform(seed = 23)),
                        Dense(128, na; initW = seed_glorot_uniform(seed = 39)),
                    ) |> cpu,
                    optimizer = ADAM(),
                ),
                batch_size = 32,
                min_replay_history = 100,
                loss_func = huber_loss,
                seed = 22,
            ),
            explorer = EpsilonGreedyExplorer(
                kind = :exp,
                ϵ_stable = 0.01,
                decay_steps = 500,
                seed = 33,
            ),
        ),
        trajectory = CircularCompactSARTSATrajectory(
            capacity = 1000,
            state_type = Float32,
            state_size = (ns,),
        ),
    )

    stop_condition = StopAfterStep(10000)

    total_reward_per_episode = TotalRewardPerEpisode()
    time_per_step = TimePerStep()
    hook = ComposedHook(
        total_reward_per_episode,
        time_per_step,
        DoEveryNStep() do t, agent, env, obs
            with_logger(lg) do
                @info "training" loss = agent.policy.learner.loss
            end
        end,
        DoEveryNEpisode() do t, agent, env, obs
            with_logger(lg) do
                @info "training" reward = total_reward_per_episode.rewards[end]
            end
        end,
        DoEveryNStep(10000) do t, agent, env, obs
            RLCore.save(save_dir, agent)
            BSON.@save joinpath(save_dir, "stats.bson") total_reward_per_episode time_per_step
        end,
    )

    description = """
    This experiment uses three dense layers to approximate the Q value.
    The testing environment is CartPoleEnv.

    Agent and statistic info will be saved to: `$save_dir`
    You can also view the tensorboard logs with `tensorboard --logdir $(joinpath(save_dir, "tb_log"))`
    To load the agent and statistic info:
    ```
    agent = RLCore.load("$save_dir", Agent)
    BSON.@load joinpath("$save_dir", "stats.bson") total_reward_per_episode time_per_step
    ```
    """

    Experiment(agent, env, stop_condition, hook, description)
end

function RLCore.Experiment(
    ::Val{:JuliaRL},
    ::Val{:DQN},
    ::Val{:CartPole},
    ::Nothing;
    save_dir = nothing,
)
    if isnothing(save_dir)
        t = Dates.format(now(), "yyyymmddHHMMSS")
        save_dir = joinpath(pwd(), "checkpoints", "JuliaRL_DQN_CartPole_$(t)")
    end

    lg = TBLogger(joinpath(save_dir, "tb_log"), min_level = Logging.Info)

    env = CartPoleEnv(; T = Float32, seed = 11)
    ns, na = length(rand(get_observation_space(env))), length(get_action_space(env))

    agent = Agent(
        policy = QBasedPolicy(
            learner = DQNLearner(
                approximator = NeuralNetworkApproximator(
                    model = Chain(
                        Dense(ns, 128, relu; initW = seed_glorot_uniform(seed = 17)),
                        Dense(128, 128, relu; initW = seed_glorot_uniform(seed = 23)),
                        Dense(128, na; initW = seed_glorot_uniform(seed = 39)),
                    ) |> cpu,
                    optimizer = ADAM(),
                ),
                target_approximator = NeuralNetworkApproximator(
                    model = Chain(
                        Dense(ns, 128, relu; initW = seed_glorot_uniform(seed = 17)),
                        Dense(128, 128, relu; initW = seed_glorot_uniform(seed = 23)),
                        Dense(128, na; initW = seed_glorot_uniform(seed = 39)),
                    ) |> cpu,
                    optimizer = ADAM(),
                ),
                loss_func = huber_loss,
                stack_size = nothing,
                batch_size = 32,
                update_horizon = 1,
                min_replay_history = 100,
                update_freq = 1,
                target_update_freq = 100,
                seed = 22,
            ),
            explorer = EpsilonGreedyExplorer(
                kind = :exp,
                ϵ_stable = 0.01,
                decay_steps = 500,
                seed = 33,
            ),
        ),
        trajectory = CircularCompactSARTSATrajectory(
            capacity = 1000,
            state_type = Float32,
            state_size = (ns,),
        ),
    )

    stop_condition = StopAfterStep(10_000)

    total_reward_per_episode = TotalRewardPerEpisode()
    time_per_step = TimePerStep()
    hook = ComposedHook(
        total_reward_per_episode,
        time_per_step,
        DoEveryNStep() do t, agent, env, obs
            if agent.policy.learner.update_step % agent.policy.learner.update_freq == 0
                with_logger(lg) do
                    @info "training" loss = agent.policy.learner.loss
                end
            end
        end,
        DoEveryNStep(10000) do t, agent, env, obs
            RLCore.save(save_dir, agent)
            BSON.@save joinpath(save_dir, "stats.bson") total_reward_per_episode time_per_step
        end,
    )

    description = """
    This experiment uses the `DQNLearner` method with three dense layers to approximate the Q value.
    The testing environment is CartPoleEnv.

    Agent and statistic info will be saved to: `$save_dir`
    You can also view the tensorboard logs with `tensorboard --logdir $(joinpath(save_dir, "tb_log"))`
    To load the agent and statistic info:

    ```
    agent = RLCore.load("$save_dir", Agent)
    BSON.@load joinpath("$save_dir", "stats.bson") total_reward_per_episode time_per_step
    ```
    """

    Experiment(agent, env, stop_condition, hook, description)
end

function RLCore.Experiment(
    ::Val{:JuliaRL},
    ::Val{:PrioritizedDQN},
    ::Val{:CartPole},
    ::Nothing;
    save_dir = nothing,
)
    if isnothing(save_dir)
        t = Dates.format(now(), "yyyymmddHHMMSS")
        save_dir = joinpath(pwd(), "checkpoints", "JuliaRL_PrioritizedDQN_CartPole_$(t)")
    end

    lg = TBLogger(joinpath(save_dir, "tb_log"), min_level = Logging.Info)

    env = CartPoleEnv(; T = Float32, seed = 11)
    ns, na = length(rand(get_observation_space(env))), length(get_action_space(env))

    agent = Agent(
        policy = QBasedPolicy(
            learner = PrioritizedDQNLearner(
                approximator = NeuralNetworkApproximator(
                    model = Chain(
                        Dense(ns, 128, relu; initW = seed_glorot_uniform(seed = 17)),
                        Dense(128, 128, relu; initW = seed_glorot_uniform(seed = 23)),
                        Dense(128, na; initW = seed_glorot_uniform(seed = 39)),
                    ) |> cpu,
                    optimizer = ADAM(),
                ),
                target_approximator = NeuralNetworkApproximator(
                    model = Chain(
                        Dense(ns, 128, relu; initW = seed_glorot_uniform(seed = 17)),
                        Dense(128, 128, relu; initW = seed_glorot_uniform(seed = 23)),
                        Dense(128, na; initW = seed_glorot_uniform(seed = 39)),
                    ) |> cpu,
                    optimizer = ADAM(),
                ),
                loss_func = huber_loss_unreduced,
                stack_size = nothing,
                batch_size = 32,
                update_horizon = 1,
                min_replay_history = 100,
                update_freq = 1,
                target_update_freq = 100,
                seed = 22,
            ),
            explorer = EpsilonGreedyExplorer(
                kind = :exp,
                ϵ_stable = 0.01,
                decay_steps = 500,
                seed = 33,
            ),
        ),
        trajectory = CircularCompactPSARTSATrajectory(
            capacity = 1000,
            state_type = Float32,
            state_size = (ns,),
        ),
    )

    stop_condition = StopAfterStep(10_000)

    total_reward_per_episode = TotalRewardPerEpisode()
    time_per_step = TimePerStep()
    hook = ComposedHook(
        total_reward_per_episode,
        time_per_step,
        DoEveryNStep() do t, agent, env, obs
            if agent.policy.learner.update_step % agent.policy.learner.update_freq == 0
                with_logger(lg) do
                    @info "training" loss = agent.policy.learner.loss
                end
            end
        end,
        DoEveryNStep(10000) do t, agent, env, obs
            RLCore.save(save_dir, agent)
            BSON.@save joinpath(save_dir, "stats.bson") total_reward_per_episode time_per_step
        end,
    )

    description = """
    This experiment uses the `PrioritizedDQNLearner` method with three dense layers to approximate the Q value.
    The testing environment is CartPoleEnv.

    Agent and statistic info will be saved to: `$save_dir`
    You can also view the tensorboard logs with `tensorboard --logdir $(joinpath(save_dir, "tb_log"))`
    To load the agent and statistic info:

    ```
    agent = RLCore.load("$save_dir", Agent)
    BSON.@load joinpath("$save_dir", "stats.bson") total_reward_per_episode time_per_step
    ```
    """

    Experiment(agent, env, stop_condition, hook, description)
end

function RLCore.Experiment(
    ::Val{:JuliaRL},
    ::Val{:Rainbow},
    ::Val{:CartPole},
    ::Nothing;
    save_dir = nothing,
)
    if isnothing(save_dir)
        t = Dates.format(now(), "yyyymmddHHMMSS")
        save_dir = joinpath(pwd(), "checkpoints", "JuliaRL_Rainbow_CartPole_$(t)")
    end

    lg = TBLogger(joinpath(save_dir, "tb_log"), min_level = Logging.Info)

    env = CartPoleEnv(; T = Float32, seed = 11)
    ns, na = length(rand(get_observation_space(env))), length(get_action_space(env))

    n_atoms = 51
    agent = Agent(
        policy = QBasedPolicy(
            learner = RainbowLearner(
                approximator = NeuralNetworkApproximator(
                    model = Chain(
                        Dense(ns, 128, relu; initW = seed_glorot_uniform(seed = 17)),
                        Dense(128, 128, relu; initW = seed_glorot_uniform(seed = 23)),
                        Dense(128, na * n_atoms; initW = seed_glorot_uniform(seed = 39)),
                    ) |> cpu,
                    optimizer = ADAM(0.0005),
                ),
                target_approximator = NeuralNetworkApproximator(
                    model = Chain(
                        Dense(ns, 128, relu; initW = seed_glorot_uniform(seed = 17)),
                        Dense(128, 128, relu; initW = seed_glorot_uniform(seed = 23)),
                        Dense(128, na * n_atoms; initW = seed_glorot_uniform(seed = 39)),
                    ) |> cpu,
                    optimizer = ADAM(0.0005),
                ),
                n_actions = na,
                n_atoms = n_atoms,
                Vₘₐₓ = 200.0f0,
                Vₘᵢₙ = 0.0f0,
                update_freq = 1,
                γ = 0.99f0,
                update_horizon = 1,
                batch_size = 32,
                stack_size = nothing,
                min_replay_history = 100,
                loss_func = logitcrossentropy_unreduced,
                target_update_freq = 100,
                seed = 22,
            ),
            explorer = EpsilonGreedyExplorer(
                kind = :exp,
                ϵ_stable = 0.01,
                decay_steps = 500,
                seed = 33,
            ),
        ),
        trajectory = CircularCompactPSARTSATrajectory(
            capacity = 1000,
            state_type = Float32,
            state_size = (ns,),
        ),
    )

    stop_condition = StopAfterStep(10_000)

    total_reward_per_episode = TotalRewardPerEpisode()
    time_per_step = TimePerStep()
    hook = ComposedHook(
        total_reward_per_episode,
        time_per_step,
        DoEveryNStep() do t, agent, env, obs
            if agent.policy.learner.update_step % agent.policy.learner.update_freq == 0
                with_logger(lg) do
                    @info "training" loss = agent.policy.learner.loss
                end
            end
        end,
        DoEveryNStep(10000) do t, agent, env, obs
            RLCore.save(save_dir, agent)
            BSON.@save joinpath(save_dir, "stats.bson") total_reward_per_episode time_per_step
        end,
    )

    description = """
    This experiment uses the `RainbowLearner` method with three dense layers to approximate the distributed Q value.
    The testing environment is CartPoleEnv.

    Agent and statistic info will be saved to: `$save_dir`
    You can also view the tensorboard logs with `tensorboard --logdir $(joinpath(save_dir, "tb_log"))`
    To load the agent and statistic info:

    ```
    agent = RLCore.load("$save_dir", Agent)
    BSON.@load joinpath("$save_dir", "stats.bson") total_reward_per_episode time_per_step
    ```
    """

    Experiment(agent, env, stop_condition, hook, description)
end

function RLCore.Experiment(
    ::Val{:JuliaRL},
    ::Val{:IQN},
    ::Val{:CartPole},
    ::Nothing;
    save_dir = nothing,
)
    if isnothing(save_dir)
        t = Dates.format(now(), "yyyymmddHHMMSS")
        save_dir = joinpath(pwd(), "checkpoints", "JuliaRL_IQN_CartPole_$(t)")
    end

    lg = TBLogger(joinpath(save_dir, "tb_log"), min_level = Logging.Info)

    env = CartPoleEnv(; T = Float32, seed = 111)
    ns, na = length(rand(get_observation_space(env))), length(get_action_space(env))

    init = seed_glorot_uniform(seed = 17)
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
                seed = 123,
                device_seed = 321,
            ),
            explorer = EpsilonGreedyExplorer(
                kind = :exp,
                ϵ_stable = 0.01,
                decay_steps = 500,
                seed = 33,
            ),
        ),
        trajectory = CircularCompactPSARTSATrajectory(
            capacity = 1000,
            state_type = Float32,
            state_size = (ns,),
        ),
    )

    stop_condition = StopAfterStep(10_000)

    total_reward_per_episode = TotalRewardPerEpisode()
    time_per_step = TimePerStep()
    hook = ComposedHook(
        total_reward_per_episode,
        time_per_step,
        DoEveryNStep() do t, agent, env, obs
            if agent.policy.learner.update_step % agent.policy.learner.update_freq == 0
                with_logger(lg) do
                    @info "training" loss = agent.policy.learner.loss
                end
            end
        end,
        DoEveryNStep(10000) do t, agent, env, obs
            RLCore.save(save_dir, agent)
            BSON.@save joinpath(save_dir, "stats.bson") total_reward_per_episode time_per_step
        end,
    )

    description = """
    This experiment uses the `IQNLearner` method with a `ImplicitQuantileNet`.
    The testing environment is CartPoleEnv.

    Agent and statistic info will be saved to: `$save_dir`
    You can also view the tensorboard logs with `tensorboard --logdir $(joinpath(save_dir, "tb_log"))`
    To load the agent and statistic info:

    ```
    agent = RLCore.load("$save_dir", Agent)
    BSON.@load joinpath("$save_dir", "stats.bson") total_reward_per_episode time_per_step
    ```
    """

    Experiment(agent, env, stop_condition, hook, description)
end

function RLCore.Experiment(
    ::Val{:JuliaRL},
    ::Val{:A2C},
    ::Val{:CartPole},
    ::Nothing;
    save_dir = nothing,
)
    N_ENV = 16
    UPDATE_FREQ = 10
    env = MultiThreadEnv([CartPoleEnv(; T = Float32, seed = i) for i in 1:N_ENV])
    ns, na = length(rand(get_observation_space(env[1]))), length(get_action_space(env[1]))
    RLBase.reset!(env, is_force = true)
    agent = Agent(
        policy = QBasedPolicy(
            learner = A2CLearner(
                approximator = ActorCritic(
                    actor = Chain(
                        Dense(ns, 256, relu; initW = seed_glorot_uniform(seed = 17)),
                        Dense(256, na; initW = seed_glorot_uniform(seed = 23)),
                    ),
                    critic = Chain(
                        Dense(ns, 256, relu; initW = seed_glorot_uniform(seed = 29)),
                        Dense(256, 1; initW = seed_glorot_uniform(seed = 29)),
                    ),
                    optimizer = ADAM(1e-3),
                ) |> cpu,
                γ = 0.99f0,
                actor_loss_weight = 1.0f0,
                critic_loss_weight = 0.5f0,
                entropy_loss_weight = 0.001f0,
            ),
            explorer = BatchExplorer(GumbelSoftmaxExplorer()),#= seed = nothing =#
        ),
        trajectory = CircularCompactSARTSATrajectory(;
            capacity = UPDATE_FREQ,
            state_type = Float32,
            state_size = (ns, N_ENV),
            action_type = Int,
            action_size = (N_ENV,),
            reward_type = Float32,
            reward_size = (N_ENV,),
            terminal_type = Bool,
            terminal_size = (N_ENV,),
        ),
    )


    hook = TotalBatchRewardPerEpisode(N_ENV)
    stop_condition = StopAfterStep(100000)
    Experiment(agent, env, stop_condition, hook, "# A2C with CartPole")
end

function RLCore.Experiment(
    ::Val{:JuliaRL},
    ::Val{:A2CGAE},
    ::Val{:CartPole},
    ::Nothing;
    save_dir = nothing,
)
    N_ENV = 16
    UPDATE_FREQ = 10
    env = MultiThreadEnv([CartPoleEnv(; T = Float32, seed = i) for i in 1:N_ENV])
    ns, na = length(rand(get_observation_space(env[1]))), length(get_action_space(env[1]))
    RLBase.reset!(env, is_force = true)
    agent = Agent(
        policy = QBasedPolicy(
            learner = A2CGAELearner(
                approximator = ActorCritic(
                    actor = NeuralNetworkApproximator(
                        model = Chain(
                            Dense(ns, 256, relu; initW = seed_glorot_uniform(seed = 17)),
                            Dense(256, na; initW = seed_glorot_uniform(seed = 23)),
                        ),
                        optimizer = ADAM(1e-3),
                    ),
                    critic = NeuralNetworkApproximator(
                        model = Chain(
                            Dense(ns, 256, relu; initW = seed_glorot_uniform(seed = 29)),
                            Dense(256, 1; initW = seed_glorot_uniform(seed = 29)),
                        ),
                        optimizer = ADAM(1e-3),
                    ),
                ) |> cpu,
                γ = 0.99f0,
                λ = 0.97f0,
                actor_loss_weight = 1.0f0,
                critic_loss_weight = 0.5f0,
                entropy_loss_weight = 0.001f0,
            ),
            explorer = BatchExplorer(GumbelSoftmaxExplorer(;)), #= seed = nothing =#
        ),
        trajectory = CircularCompactSARTSATrajectory(;
            capacity = UPDATE_FREQ,
            state_type = Float32,
            state_size = (ns, N_ENV),
            action_type = Int,
            action_size = (N_ENV,),
            reward_type = Float32,
            reward_size = (N_ENV,),
            terminal_type = Bool,
            terminal_size = (N_ENV,),
        ),
    )
    Experiment(
        agent,
        env,
        StopAfterStep(100000),
        TotalBatchRewardPerEpisode(N_ENV),
        "# A2CGAE with CartPole",
    )
end

function RLCore.Experiment(::Val{:JuliaRL}, ::Val{:DDPG}, ::Val{:Pendulum}, ::Nothing;)
    inner_env = PendulumEnv(T = Float32, seed = 9231)
    action_space = get_action_space(inner_env)
    low = action_space.low
    high = action_space.high
    ns = length(rand(get_observation_space(inner_env)))

    env = WrappedEnv(;
        env = inner_env,
        postprocessor = ((x,),) -> (low + (x + 1) * 0.5 * (high - low),), # rescale [-1, 1] -> (low, high)
    )

    init = seed_glorot_uniform(seed = 17)

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
            start_policy = RandomPolicy(ContinuousSpace(-1.0, 1.0); seed = 923),
            update_after = 1000,
            update_every = 1,
            act_limit = 1.0,
            act_noise = 0.1,
            seed = 131,
        ),
        trajectory = CircularCompactSARTSATrajectory(
            capacity = 10000,
            state_type = Float32,
            state_size = (ns,),
            action_type = Float32,
        ),
    )

    description = """
    # Play Pendulum with DDPG
    """

    Experiment(agent, env, StopAfterStep(10000), TotalRewardPerEpisode(), description)
end

function RLCore.Experiment(::Val{:JuliaRL}, ::Val{:PPO}, ::Val{:CartPole}, ::Nothing;)
    N_ENV = 8
    UPDATE_FREQ = 16
    env = MultiThreadEnv([CartPoleEnv(; T = Float32, seed = i) for i in 1:N_ENV])
    ns, na = length(rand(get_observation_space(env[1]))), length(get_action_space(env[1]))
    RLBase.reset!(env, is_force = true)
    agent = Agent(
        policy = QBasedPolicy(
            learner = PPOLearner(
                approximator = ActorCritic(
                    actor = NeuralNetworkApproximator(
                        model = Chain(
                            Dense(ns, 256, relu; initW = seed_glorot_uniform(seed = 17)),
                            Dense(256, na; initW = seed_glorot_uniform(seed = 23)),
                        ),
                        optimizer = ADAM(1e-3),
                    ),
                    critic = NeuralNetworkApproximator(
                        model = Chain(
                            Dense(ns, 256, relu; initW = seed_glorot_uniform(seed = 29)),
                            Dense(256, 1; initW = seed_glorot_uniform(seed = 29)),
                        ),
                        optimizer = ADAM(1e-3),
                    ),
                ) |> cpu,
                γ = 0.99f0,
                λ = 0.95f0,
                clip_range = 0.1f0,
                max_grad_norm = 0.5f0,
                n_epochs = 4,
                n_microbatches = 4,
                actor_loss_weight = 1.0f0,
                critic_loss_weight = 0.5f0,
                entropy_loss_weight = 0.001f0,
            ),
            explorer = BatchExplorer(GumbelSoftmaxExplorer(; seed = 1)),
        ),
        trajectory = PPOTrajectory(;
            capacity = 32,
            state_type = Float32,
            state_size = (ns, N_ENV),
            action_type = Int,
            action_size = (N_ENV,),
            action_log_prob_type = Float32,
            action_log_prob_size = (N_ENV,),
            reward_type = Float32,
            reward_size = (N_ENV,),
            terminal_type = Bool,
            terminal_size = (N_ENV,),
        ),
    )
    Experiment(
        agent,
        env,
        StopAfterStep(100000),
        TotalBatchRewardPerEpisode(N_ENV),
        "# PPO with CartPole",
    )
end

function RLCore.Experiment(
    ::Val{:JuliaRL},
    ::Val{:BasicDQN},
    ::Val{:MountainCar},
    ::Nothing;
    save_dir = nothing,
)
    if isnothing(save_dir)
        t = Dates.format(now(), "yyyymmddHHMMSS")
        save_dir = joinpath(pwd(), "checkpoints", "JuliaRL_BasicDQN_MountainCar_$(t)")
    end

    lg = TBLogger(joinpath(save_dir, "tb_log"), min_level = Logging.Info)

    env = MountainCarEnv(; T = Float32, max_steps = 5000, seed = 11)
    ns, na = length(rand(get_observation_space(env))), length(get_action_space(env))
    agent = Agent(
        policy = QBasedPolicy(
            learner = BasicDQNLearner(
                approximator = NeuralNetworkApproximator(
                    model = Chain(
                        Dense(ns, 64, relu; initW = seed_glorot_uniform(seed = 17)),
                        Dense(64, 64, relu; initW = seed_glorot_uniform(seed = 23)),
                        Dense(64, na; initW = seed_glorot_uniform(seed = 39)),
                    ) |> cpu,
                    optimizer = ADAM(),
                ),
                batch_size = 32,
                min_replay_history = 100,
                loss_func = huber_loss,
                seed = 22,
            ),
            explorer = EpsilonGreedyExplorer(
                kind = :exp,
                ϵ_stable = 0.01,
                decay_steps = 500,
                seed = 33,
            ),
        ),
        trajectory = CircularCompactSARTSATrajectory(
            capacity = 50000,
            state_type = Float32,
            state_size = (ns,),
        ),
    )

    stop_condition = StopAfterStep(40000)

    total_reward_per_episode = TotalRewardPerEpisode()
    time_per_step = TimePerStep()
    hook = ComposedHook(
        total_reward_per_episode,
        time_per_step,
        DoEveryNStep() do t, agent, env, obs
            with_logger(lg) do
                @info "training" loss = agent.policy.learner.loss
            end
        end,
        DoEveryNEpisode() do t, agent, env, obs
            with_logger(lg) do
                @info "training" reward = total_reward_per_episode.rewards[end]
            end
        end,
        DoEveryNStep(10000) do t, agent, env, obs
            RLCore.save(save_dir, agent)
            BSON.@save joinpath(save_dir, "stats.bson") total_reward_per_episode time_per_step
        end,
    )

    description = """
    This experiment uses three dense layers to approximate the Q value.
    The testing environment is MountainCarEnv.

    Agent and statistic info will be saved to: `$save_dir`
    You can also view the tensorboard logs with `tensorboard --logdir $(joinpath(save_dir, "tb_log"))`
    To load the agent and statistic info:
    ```
    agent = RLCore.load("$save_dir", Agent)
    BSON.@load joinpath("$save_dir", "stats.bson") total_reward_per_episode time_per_step
    ```
    """

    Experiment(agent, env, stop_condition, hook, description)
end

function RLCore.Experiment(
    ::Val{:JuliaRL},
    ::Val{:DQN},
    ::Val{:MountainCar},
    ::Nothing;
    save_dir = nothing,
)
    if isnothing(save_dir)
        t = Dates.format(now(), "yyyymmddHHMMSS")
        save_dir = joinpath(pwd(), "checkpoints", "JuliaRL_DQN_MountainCar_$(t)")
    end

    lg = TBLogger(joinpath(save_dir, "tb_log"), min_level = Logging.Info)

    env = MountainCarEnv(; T = Float32, max_steps = 5000, seed = 11)
    ns, na = length(rand(get_observation_space(env))), length(get_action_space(env))

    agent = Agent(
        policy = QBasedPolicy(
            learner = DQNLearner(
                approximator = NeuralNetworkApproximator(
                    model = Chain(
                        Dense(ns, 64, relu; initW = seed_glorot_uniform(seed = 17)),
                        Dense(64, 64, relu; initW = seed_glorot_uniform(seed = 23)),
                        Dense(64, na; initW = seed_glorot_uniform(seed = 39)),
                    ) |> cpu,
                    optimizer = ADAM(),
                ),
                target_approximator = NeuralNetworkApproximator(
                    model = Chain(
                        Dense(ns, 64, relu; initW = seed_glorot_uniform(seed = 17)),
                        Dense(64, 64, relu; initW = seed_glorot_uniform(seed = 23)),
                        Dense(64, na; initW = seed_glorot_uniform(seed = 39)),
                    ) |> cpu,
                    optimizer = ADAM(),
                ),
                loss_func = huber_loss,
                stack_size = nothing,
                batch_size = 32,
                update_horizon = 1,
                min_replay_history = 100,
                update_freq = 1,
                target_update_freq = 100,
                seed = 22,
            ),
            explorer = EpsilonGreedyExplorer(
                kind = :exp,
                ϵ_stable = 0.01,
                decay_steps = 500,
                seed = 33,
            ),
        ),
        trajectory = CircularCompactSARTSATrajectory(
            capacity = 50000,
            state_type = Float32,
            state_size = (ns,),
        ),
    )

    stop_condition = StopAfterStep(40_000)

    total_reward_per_episode = TotalRewardPerEpisode()
    time_per_step = TimePerStep()
    hook = ComposedHook(
        total_reward_per_episode,
        time_per_step,
        DoEveryNStep() do t, agent, env, obs
            if agent.policy.learner.update_step % agent.policy.learner.update_freq == 0
                with_logger(lg) do
                    @info "training" loss = agent.policy.learner.loss
                end
            end
        end,
        DoEveryNStep(10000) do t, agent, env, obs
            RLCore.save(save_dir, agent)
            BSON.@save joinpath(save_dir, "stats.bson") total_reward_per_episode time_per_step
        end,
    )

    description = """
    This experiment uses the `DQNLearner` method with three dense layers to approximate the Q value.
    The testing environment is MountainCarEnv.

    Agent and statistic info will be saved to: `$save_dir`
    You can also view the tensorboard logs with `tensorboard --logdir $(joinpath(save_dir, "tb_log"))`
    To load the agent and statistic info:

    ```
    agent = RLCore.load("$save_dir", Agent)
    BSON.@load joinpath("$save_dir", "stats.bson") total_reward_per_episode time_per_step
    ```
    """

    Experiment(agent, env, stop_condition, hook, description)
end
