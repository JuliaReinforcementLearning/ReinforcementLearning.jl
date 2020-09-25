export Experiment

using Dates
using ReinforcementLearningBase
using ReinforcementLearningCore
using .ReinforcementLearningEnvironments
using Flux
using BSON
using TensorBoardLogger
using Logging
using Random
using Distributions: Categorical, Normal

function Description(prelude::String, save_dir::String)
    """
    $prelude

    Agent and statistic info will be saved to: `$save_dir`
    You can also view the tensorboard logs with
    `tensorboard --logdir $(joinpath(save_dir, "tb_log"))`
    To load the agent and statistic info:
    ```
    agent = RLCore.load("$save_dir", Agent)
    BSON.@load joinpath("$save_dir", "stats.bson") total_reward_per_episode time_per_step
    ```
"""
end

function RLCore.Experiment(
    ::Val{:JuliaRL},
    ::Val{:BasicDQN},
    ::Val{:CartPole},
    ::Nothing;
    save_dir = nothing,
    seed = 123,
)
    if isnothing(save_dir)
        t = Dates.format(now(), "yyyy_mm_dd_HH_MM_SS")
        save_dir = joinpath(pwd(), "checkpoints", "JuliaRL_BasicDQN_CartPole_$(t)")
    end

    lg = TBLogger(joinpath(save_dir, "tb_log"), min_level = Logging.Info)
    rng = MersenneTwister(seed)

    env = CartPoleEnv(; T = Float32, rng = rng)
    ns, na = length(get_state(env)), length(get_actions(env))
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
        DoEveryNStep(10000) do t, agent, env
            RLCore.save(save_dir, agent)
            BSON.@save joinpath(save_dir, "stats.bson") total_reward_per_episode time_per_step
        end,
    )

    description = """
    This experiment uses three dense layers to approximate the Q value.
    The testing environment is CartPoleEnv.
    """

    Experiment(agent, env, stop_condition, hook, Description(description, save_dir))
end

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
    rng = MersenneTwister(seed)

    env = CartPoleEnv(; T = Float32, rng = rng)
    ns, na = length(get_state(env)), length(get_actions(env))

    agent = Agent(
        policy = QBasedPolicy(
            learner = DQNLearner(
                approximator = NeuralNetworkApproximator(
                    model = Chain(
                        Dense(ns, 128, relu; initW = glorot_uniform(rng)),
                        Dense(128, 128, relu; initW = glorot_uniform(rng)),
                        Dense(128, na; initW = glorot_uniform(rng)),
                    ) |> cpu,
                    optimizer = ADAM(),
                ),
                target_approximator = NeuralNetworkApproximator(
                    model = Chain(
                        Dense(ns, 128, relu; initW = glorot_uniform(rng)),
                        Dense(128, 128, relu; initW = glorot_uniform(rng)),
                        Dense(128, na; initW = glorot_uniform(rng)),
                    ) |> cpu,
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
        DoEveryNStep(10000) do t, agent, env
            RLCore.save(save_dir, agent)
            BSON.@save joinpath(save_dir, "stats.bson") total_reward_per_episode time_per_step
        end,
    )

    description = """
    This experiment uses the `DQNLearner` method with three dense layers to approximate the Q value.
    The testing environment is CartPoleEnv.
    """

    Experiment(agent, env, stop_condition, hook, Description(description, save_dir))
end

function RLCore.Experiment(
    ::Val{:JuliaRL},
    ::Val{:PrioritizedDQN},
    ::Val{:CartPole},
    ::Nothing;
    save_dir = nothing,
    seed = 123,
)
    if isnothing(save_dir)
        t = Dates.format(now(), "yyyy_mm_dd_HH_MM_SS")
        save_dir = joinpath(pwd(), "checkpoints", "JuliaRL_PrioritizedDQN_CartPole_$(t)")
    end

    lg = TBLogger(joinpath(save_dir, "tb_log"), min_level = Logging.Info)
    rng = MersenneTwister(seed)

    env = CartPoleEnv(; T = Float32, rng = rng)
    ns, na = length(get_state(env)), length(get_actions(env))

    agent = Agent(
        policy = QBasedPolicy(
            learner = PrioritizedDQNLearner(
                approximator = NeuralNetworkApproximator(
                    model = Chain(
                        Dense(ns, 128, relu; initW = glorot_uniform(rng)),
                        Dense(128, 128, relu; initW = glorot_uniform(rng)),
                        Dense(128, na; initW = glorot_uniform(rng)),
                    ) |> cpu,
                    optimizer = ADAM(),
                ),
                target_approximator = NeuralNetworkApproximator(
                    model = Chain(
                        Dense(ns, 128, relu; initW = glorot_uniform(rng)),
                        Dense(128, 128, relu; initW = glorot_uniform(rng)),
                        Dense(128, na; initW = glorot_uniform(rng)),
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
                rng = rng,
            ),
            explorer = EpsilonGreedyExplorer(
                kind = :exp,
                ϵ_stable = 0.01,
                decay_steps = 500,
                rng = rng,
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
        DoEveryNStep(10000) do t, agent, env
            RLCore.save(save_dir, agent)
            BSON.@save joinpath(save_dir, "stats.bson") total_reward_per_episode time_per_step
        end,
    )

    description = """
    This experiment uses the `PrioritizedDQNLearner` method with three dense layers to approximate the Q value.
    The testing environment is CartPoleEnv.
    """

    Experiment(agent, env, stop_condition, hook, Description(description, save_dir))
end

function RLCore.Experiment(
    ::Val{:JuliaRL},
    ::Val{:Rainbow},
    ::Val{:CartPole},
    ::Nothing;
    save_dir = nothing,
    seed = 123,
)
    if isnothing(save_dir)
        t = Dates.format(now(), "yyyy_mm_dd_HH_MM_SS")
        save_dir = joinpath(pwd(), "checkpoints", "JuliaRL_Rainbow_CartPole_$(t)")
    end

    lg = TBLogger(joinpath(save_dir, "tb_log"), min_level = Logging.Info)
    rng = MersenneTwister(seed)

    env = CartPoleEnv(; T = Float32, rng = rng)
    ns, na = length(get_state(env)), length(get_actions(env))

    n_atoms = 51
    agent = Agent(
        policy = QBasedPolicy(
            learner = RainbowLearner(
                approximator = NeuralNetworkApproximator(
                    model = Chain(
                        Dense(ns, 128, relu; initW = glorot_uniform(rng)),
                        Dense(128, 128, relu; initW = glorot_uniform(rng)),
                        Dense(128, na * n_atoms; initW = glorot_uniform(rng)),
                    ) |> cpu,
                    optimizer = ADAM(0.0005),
                ),
                target_approximator = NeuralNetworkApproximator(
                    model = Chain(
                        Dense(ns, 128, relu; initW = glorot_uniform(rng)),
                        Dense(128, 128, relu; initW = glorot_uniform(rng)),
                        Dense(128, na * n_atoms; initW = glorot_uniform(rng)),
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
                rng = rng,
            ),
            explorer = EpsilonGreedyExplorer(
                kind = :exp,
                ϵ_stable = 0.01,
                decay_steps = 500,
                rng = rng,
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
        DoEveryNStep(10000) do t, agent, env
            RLCore.save(save_dir, agent)
            BSON.@save joinpath(save_dir, "stats.bson") total_reward_per_episode time_per_step
        end,
    )

    description = """
    This experiment uses the `RainbowLearner` method with three dense layers to approximate the distributed Q value.
    The testing environment is CartPoleEnv.
    """

    Experiment(agent, env, stop_condition, hook, Description(description, save_dir))
end

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
    rng = MersenneTwister(seed)

    env = CartPoleEnv(; T = Float32, rng = rng)
    ns, na = length(get_state(env)), length(get_actions(env))

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
        DoEveryNStep(10000) do t, agent, env
            RLCore.save(save_dir, agent)
            BSON.@save joinpath(save_dir, "stats.bson") total_reward_per_episode time_per_step
        end,
    )

    description = """
    This experiment uses the `IQNLearner` method with a `ImplicitQuantileNet`.
    The testing environment is CartPoleEnv.
    """

    Experiment(agent, env, stop_condition, hook, Description(description, save_dir))
end

function RLCore.Experiment(
    ::Val{:JuliaRL},
    ::Val{:A2C},
    ::Val{:CartPole},
    ::Nothing;
    save_dir = nothing,
    seed = 123,
)
    if isnothing(save_dir)
        t = Dates.format(now(), "yyyy_mm_dd_HH_MM_SS")
        save_dir = joinpath(pwd(), "checkpoints", "JuliaRL_A2C_CartPole_$(t)")
    end

    lg = TBLogger(joinpath(save_dir, "tb_log"), min_level = Logging.Info)
    rng = MersenneTwister(seed)
    N_ENV = 16
    UPDATE_FREQ = 10
    env = MultiThreadEnv([
        CartPoleEnv(; T = Float32, rng = MersenneTwister(hash(seed + i))) for i in 1:N_ENV
    ])
    ns, na = length(get_state(env[1])), length(get_actions(env[1]))
    RLBase.reset!(env, is_force = true)
    agent = Agent(
        policy = QBasedPolicy(
            learner = A2CLearner(
                approximator = ActorCritic(
                    actor = Chain(
                        Dense(ns, 256, relu; initW = glorot_uniform(rng)),
                        Dense(256, na; initW = glorot_uniform(rng)),
                    ),
                    critic = Chain(
                        Dense(ns, 256, relu; initW = glorot_uniform(rng)),
                        Dense(256, 1; initW = glorot_uniform(rng)),
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

    stop_condition = StopAfterStep(haskey(ENV, "CI") ? 10_000 : 100_000)
    total_reward_per_episode = TotalBatchRewardPerEpisode(N_ENV)
    time_per_step = TimePerStep()
    hook = ComposedHook(
        total_reward_per_episode,
        time_per_step,
        DoEveryNStep() do t, agent, env
            with_logger(lg) do
                @info(
                    "training",
                    actor_loss = agent.policy.learner.actor_loss,
                    critic_loss = agent.policy.learner.critic_loss,
                    entropy_loss = agent.policy.learner.entropy_loss,
                    loss = agent.policy.learner.loss,
                )
                for i in 1:length(env)
                    if get_terminal(env[i])
                        @info "training" reward = total_reward_per_episode.rewards[i][end] log_step_increment =
                            0
                        break
                    end
                end
            end
        end,
        DoEveryNStep(10000) do t, agent, env
            RLCore.save(save_dir, agent)
            BSON.@save joinpath(save_dir, "stats.bson") total_reward_per_episode time_per_step
        end,
    )
    Experiment(
        agent,
        env,
        stop_condition,
        hook,
        Description("# A2C with CartPole", save_dir),
    )
end

function RLCore.Experiment(
    ::Val{:JuliaRL},
    ::Val{:A2CGAE},
    ::Val{:CartPole},
    ::Nothing;
    save_dir = nothing,
    seed = 123,
)
    if isnothing(save_dir)
        t = Dates.format(now(), "yyyy_mm_dd_HH_MM_SS")
        save_dir = joinpath(pwd(), "checkpoints", "JuliaRL_A2CGAE_CartPole_$(t)")
    end

    lg = TBLogger(joinpath(save_dir, "tb_log"), min_level = Logging.Info)
    rng = MersenneTwister(seed)
    N_ENV = 16
    UPDATE_FREQ = 10
    env = MultiThreadEnv([
        CartPoleEnv(; T = Float32, rng = MersenneTwister(hash(seed + i))) for i in 1:N_ENV
    ])
    ns, na = length(get_state(env[1])), length(get_actions(env[1]))
    RLBase.reset!(env, is_force = true)
    agent = Agent(
        policy = QBasedPolicy(
            learner = A2CGAELearner(
                approximator = ActorCritic(
                    actor = NeuralNetworkApproximator(
                        model = Chain(
                            Dense(ns, 256, relu; initW = glorot_uniform(rng)),
                            Dense(256, na; initW = glorot_uniform(rng)),
                        ),
                        optimizer = ADAM(1e-3),
                    ),
                    critic = NeuralNetworkApproximator(
                        model = Chain(
                            Dense(ns, 256, relu; initW = glorot_uniform(rng)),
                            Dense(256, 1; initW = glorot_uniform(rng)),
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
    stop_condition = StopAfterStep(haskey(ENV, "CI") ? 10_000 : 100_000)
    total_reward_per_episode = TotalBatchRewardPerEpisode(N_ENV)
    time_per_step = TimePerStep()
    hook = ComposedHook(
        total_reward_per_episode,
        time_per_step,
        DoEveryNStep() do t, agent, env
            with_logger(lg) do
                @info(
                    "training",
                    actor_loss = agent.policy.learner.actor_loss,
                    critic_loss = agent.policy.learner.critic_loss,
                    entropy_loss = agent.policy.learner.entropy_loss,
                    loss = agent.policy.learner.loss,
                )
                for i in 1:length(env)
                    if get_terminal(env[i])
                        @info "training" reward = total_reward_per_episode.rewards[i][end] log_step_increment =
                            0
                        break
                    end
                end
            end
        end,
        DoEveryNStep(10000) do t, agent, env
            RLCore.save(save_dir, agent)
            BSON.@save joinpath(save_dir, "stats.bson") total_reward_per_episode time_per_step
        end,
    )
    Experiment(
        agent,
        env,
        stop_condition,
        hook,
        Description("# A2CGAE with CartPole", save_dir),
    )
end

function RLCore.Experiment(
    ::Val{:JuliaRL},
    ::Val{:DDPG},
    ::Val{:Pendulum},
    ::Nothing;
    save_dir = nothing,
    seed = 123,
)
    if isnothing(save_dir)
        t = Dates.format(now(), "yyyy_mm_dd_HH_MM_SS")
        save_dir = joinpath(pwd(), "checkpoints", "JuliaRL_DDPG_Pendulum_$(t)")
    end

    lg = TBLogger(joinpath(save_dir, "tb_log"), min_level = Logging.Info)
    rng = MersenneTwister(seed)
    inner_env = PendulumEnv(T = Float32, rng = rng)
    action_space = get_actions(inner_env)
    low = action_space.low
    high = action_space.high
    ns = length(get_state(inner_env))

    env = inner_env |> ActionTransformedEnv(x -> low + (x + 1) * 0.5 * (high - low))
    init = glorot_uniform(rng)

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
            start_policy = RandomPolicy(ContinuousSpace(-1.0, 1.0); rng = rng),
            update_after = 1000,
            update_every = 1,
            act_limit = 1.0,
            act_noise = 0.1,
            rng = rng,
        ),
        trajectory = CircularCompactSARTSATrajectory(
            capacity = 10000,
            state_type = Float32,
            state_size = (ns,),
            action_type = Float32,
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
        DoEveryNStep(10000) do t, agent, env
            RLCore.save(save_dir, agent)
            BSON.@save joinpath(save_dir, "stats.bson") total_reward_per_episode time_per_step
        end,
    )

    Experiment(
        agent,
        env,
        stop_condition,
        hook,
        Description("# Play Pendulum with DDPG", save_dir),
    )
end

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
    rng = MersenneTwister(seed)
    inner_env = PendulumEnv(T = Float32, rng = rng)
    action_space = get_actions(inner_env)
    low = action_space.low
    high = action_space.high
    ns = length(get_state(inner_env))

    env = inner_env |> ActionTransformedEnv(x -> low + (x + 1) * 0.5 * (high - low))
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
            start_policy = RandomPolicy(ContinuousSpace(-1.0, 1.0); rng = rng),
            update_after = 1000,
            update_every = 1,
            policy_freq = 2,
            target_act_limit = 1.0,
            target_act_noise = 0.1,
            act_limit = 1.0,
            act_noise = 0.1,
            rng = rng,
        ),
        trajectory = CircularCompactSARTSATrajectory(
            capacity = 10000,
            state_type = Float32,
            state_size = (ns,),
            action_type = Float32,
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
        DoEveryNStep(10000) do t, agent, env
            RLCore.save(save_dir, agent)
            BSON.@save joinpath(save_dir, "stats.bson") total_reward_per_episode time_per_step
        end,
    )

    Experiment(
        agent,
        env,
        stop_condition,
        hook,
        Description("# Play Pendulum with TD3", save_dir),
    )
end

function RLCore.Experiment(
    ::Val{:JuliaRL},
    ::Val{:PPO},
    ::Val{:CartPole},
    ::Nothing;
    save_dir = nothing,
    seed = 123,
)
    if isnothing(save_dir)
        t = Dates.format(now(), "yyyy_mm_dd_HH_MM_SS")
        save_dir = joinpath(pwd(), "checkpoints", "JuliaRL_PPO_CartPole_$(t)")
    end

    lg = TBLogger(joinpath(save_dir, "tb_log"), min_level = Logging.Info)
    rng = MersenneTwister(seed)
    N_ENV = 8
    UPDATE_FREQ = 16
    env = MultiThreadEnv([
        CartPoleEnv(; T = Float32, rng = MersenneTwister(hash(seed + i))) for i in 1:N_ENV
    ])
    ns, na = length(get_state(env[1])), length(get_actions(env[1]))
    RLBase.reset!(env, is_force = true)
    agent = Agent(
        policy = QBasedPolicy(
            learner = PPOLearner(
                approximator = ActorCritic(
                    actor = NeuralNetworkApproximator(
                        model = Chain(
                            Dense(ns, 256, relu; initW = glorot_uniform(rng)),
                            Dense(256, na; initW = glorot_uniform(rng)),
                        ),
                        optimizer = ADAM(1e-3),
                    ),
                    critic = NeuralNetworkApproximator(
                        model = Chain(
                            Dense(ns, 256, relu; initW = glorot_uniform(rng)),
                            Dense(256, 1; initW = glorot_uniform(rng)),
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
            explorer = BatchExplorer(GumbelSoftmaxExplorer(; rng = rng)),
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

    stop_condition = StopAfterStep(haskey(ENV, "CI") ? 10_000 : 100_000)
    total_reward_per_episode = TotalBatchRewardPerEpisode(N_ENV)
    time_per_step = TimePerStep()
    hook = ComposedHook(
        total_reward_per_episode,
        time_per_step,
        DoEveryNStep() do t, agent, env
            with_logger(lg) do
                @info(
                    "training",
                    actor_loss = agent.policy.learner.actor_loss[end, end],
                    critic_loss = agent.policy.learner.critic_loss[end, end],
                    loss = agent.policy.learner.loss[end, end],
                )
                for i in 1:length(env)
                    if get_terminal(env[i])
                        @info "training" reward = total_reward_per_episode.rewards[i][end] log_step_increment =
                            0
                        break
                    end
                end
            end
        end,
        DoEveryNStep(10000) do t, agent, env
            RLCore.save(save_dir, agent)
            BSON.@save joinpath(save_dir, "stats.bson") total_reward_per_episode time_per_step
        end,
    )

    Experiment(
        agent,
        env,
        stop_condition,
        hook,
        Description("# PPO with CartPole", save_dir),
    )
end

function RLCore.Experiment(
    ::Val{:JuliaRL},
    ::Val{:BasicDQN},
    ::Val{:MountainCar},
    ::Nothing;
    save_dir = nothing,
    seed = 123,
)
    rng = MersenneTwister(seed)
    if isnothing(save_dir)
        t = Dates.format(now(), "yyyy_mm_dd_HH_MM_SS")
        save_dir = joinpath(pwd(), "checkpoints", "JuliaRL_BasicDQN_MountainCar_$(t)")
    end

    lg = TBLogger(joinpath(save_dir, "tb_log"), min_level = Logging.Info)

    env = MountainCarEnv(; T = Float32, max_steps = 5000, rng = rng)
    ns, na = length(get_state(env)), length(get_actions(env))
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
        trajectory = CircularCompactSARTSATrajectory(
            capacity = 50000,
            state_type = Float32,
            state_size = (ns,),
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
        DoEveryNStep(10000) do t, agent, env
            RLCore.save(save_dir, agent)
            BSON.@save joinpath(save_dir, "stats.bson") total_reward_per_episode time_per_step
        end,
    )

    description = """
    This experiment uses three dense layers to approximate the Q value.
    The testing environment is MountainCarEnv.
    """

    Experiment(agent, env, stop_condition, hook, Description(description, save_dir))
end

function RLCore.Experiment(
    ::Val{:JuliaRL},
    ::Val{:DQN},
    ::Val{:MountainCar},
    ::Nothing;
    save_dir = nothing,
    seed = 123,
)
    if isnothing(save_dir)
        t = Dates.format(now(), "yyyy_mm_dd_HH_MM_SS")
        save_dir = joinpath(pwd(), "checkpoints", "JuliaRL_DQN_MountainCar_$(t)")
    end

    lg = TBLogger(joinpath(save_dir, "tb_log"), min_level = Logging.Info)
    rng = MersenneTwister(seed)

    env = MountainCarEnv(; T = Float32, max_steps = 5000, rng = rng)
    ns, na = length(get_state(env)), length(get_actions(env))

    agent = Agent(
        policy = QBasedPolicy(
            learner = DQNLearner(
                approximator = NeuralNetworkApproximator(
                    model = Chain(
                        Dense(ns, 64, relu; initW = glorot_uniform(rng)),
                        Dense(64, 64, relu; initW = glorot_uniform(rng)),
                        Dense(64, na; initW = glorot_uniform(rng)),
                    ) |> cpu,
                    optimizer = ADAM(),
                ),
                target_approximator = NeuralNetworkApproximator(
                    model = Chain(
                        Dense(ns, 64, relu; initW = glorot_uniform(rng)),
                        Dense(64, 64, relu; initW = glorot_uniform(rng)),
                        Dense(64, na; initW = glorot_uniform(rng)),
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
                rng = rng,
            ),
            explorer = EpsilonGreedyExplorer(
                kind = :exp,
                ϵ_stable = 0.01,
                decay_steps = 500,
                rng = rng,
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
        DoEveryNStep(10000) do t, agent, env
            RLCore.save(save_dir, agent)
            BSON.@save joinpath(save_dir, "stats.bson") total_reward_per_episode time_per_step
        end,
    )

    description = """
    This experiment uses the `DQNLearner` method with three dense layers to approximate the Q value.
    The testing environment is MountainCarEnv.
    """

    Experiment(agent, env, stop_condition, hook, Description(description, save_dir))
end

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
    rng = MersenneTwister(seed)
    inner_env = PendulumEnv(T = Float32, rng = rng)
    action_space = get_actions(inner_env)
    low = action_space.low
    high = action_space.high
    ns = length(get_state(inner_env))

    env = inner_env |> ActionTransformedEnv(x -> low + (x + 1) * 0.5 * (high - low))
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
            start_policy = RandomPolicy(ContinuousSpace(-1.0, 1.0); rng = rng),
            update_after = 1000,
            update_every = 1,
            rng = rng,
        ),
        trajectory = CircularCompactSARTSATrajectory(
            capacity = 10000,
            state_type = Float32,
            state_size = (ns,),
            action_type = Float32,
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
        DoEveryNStep(10000) do t, agent, env
            RLCore.save(save_dir, agent)
            BSON.@save joinpath(save_dir, "stats.bson") total_reward_per_episode time_per_step
        end,
    )

    Experiment(
        agent,
        env,
        stop_condition,
        hook,
        Description("# Play Pendulum with SAC", save_dir),
    )
end

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
    rng = MersenneTwister(seed)
    env = CartPoleEnv(; T = Float32, rng = rng)
    ns, na = length(get_state(env)), length(get_actions(env))

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
            action_space = get_actions(env),
            dist = Categorical,
            γ = 0.99f0,
            rng = rng,
        ),
        trajectory = ElasticCompactSARTSATrajectory(
            state_type = Float32,
            state_size = (ns,),
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
        DoEveryNEpisode(500) do t, agent, env
            RLCore.save(save_dir, agent)
            BSON.@save joinpath(save_dir, "stats.bson") total_reward_per_episode time_per_step
        end,
    )

    description = Description("# Play CartPole with VPG", save_dir)

    Experiment(agent, env, stop_condition, hook, description)
end

function RLCore.Experiment(
    ::Val{:JuliaRL},
    ::Val{:VPG},
    ::Val{:PendulumD},
    ::Nothing;
    save_dir = nothing,
    seed = 2213,
)
    if isnothing(save_dir)
        t = Dates.format(now(), "yyyy_mm_dd_HH_MM_SS")
        save_dir = joinpath(pwd(), "checkpoints", "JuliaRL_VPG_PendulumD_$(t)")
    end

    lg = TBLogger(joinpath(save_dir, "tb_log"), min_level = Logging.Info)
    rng = MersenneTwister(seed)
    env = PendulumEnv(;
        T = Float32,
        rng = rng,
        continuous = false,
        n_actions = 3,
        max_steps = 100,
    )
    ns, na = length(get_state(env)), length(get_actions(env))

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
            action_space = get_actions(env),
            dist = Categorical,
            γ = 0.99f0,
            rng = rng,
        ),
        trajectory = ElasticCompactSARTSATrajectory(
            state_type = Float32,
            state_size = (ns,),
        ),
    )

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
        DoEveryNEpisode(5000) do t, agent, env
            RLCore.save(save_dir, agent)
            BSON.@save joinpath(save_dir, "stats.bson") total_reward_per_episode time_per_step
        end,
    )

    Experiment(
        agent,
        env,
        StopAfterEpisode(3000),
        hook,
        Description("# Play Pendulum(Discrete) with VPG", save_dir),
    )
end

function RLCore.Experiment(
    ::Val{:JuliaRL},
    ::Val{:VPG},
    ::Val{:Pendulum},
    ::Nothing;
    save_dir = nothing,
    seed = 5574,
)
    #= TODO:
    This only acts as a template implementation for the vpg in a continuous action space.
    Due to it doesn't converge in most cases. it only works with a few random seeds.
    I'm not sure if it's the limitation of VPG, or I used wrong hyper parameters.
    =#
    if isnothing(save_dir)
        t = Dates.format(now(), "yyyy_mm_dd_HH_MM_SS")
        save_dir = joinpath(pwd(), "checkpoints", "JuliaRL_VPG_Pendulum_$(t)")
    end

    lg = TBLogger(joinpath(save_dir, "tb_log"), min_level = Logging.Info)
    rng = MersenneTwister(seed)

    inner_env = PendulumEnv(; T = Float32, rng = rng, max_steps = 100)
    high, low = get_actions(inner_env) |> x -> (x.low, x.high)
    TransformAction(A) = tanh(A) * 0.5f0 * (high - low) + (high + low) / 2 # tanh's output ∈ [-1,1]
    env = inner_env |> ActionTransformedEnv(TransformAction)
    ns = length(get_state(env))

    agent = Agent(
        policy = VPGPolicy(
            approximator = NeuralNetworkApproximator(
                model = GaussianNetwork(
                    Chain(
                        Dense(ns, 128, relu, initW = glorot_uniform(rng)),
                        Dense(128, 128, relu, initW = glorot_uniform(rng)),
                    ),
                    Chain(Dense(128, 1, x -> 2 * tanh(x), initW = glorot_uniform(rng))), # limit the range of μ in [-2,2]
                    Chain(x -> -0.5), # use a fixed σ. its too difficult to learn 2 parameters at the same time.
                ),
                optimizer = ADAM(0.001),
            ) |> cpu,
            baseline = NeuralNetworkApproximator(
                model = Chain(
                    Dense(ns, 128, relu; initW = glorot_uniform(rng)),
                    Dense(128, 128, relu; initW = glorot_uniform(rng)),
                    Dense(128, 1; initW = glorot_uniform(rng)),
                ),
                optimizer = ADAM(),
            ) |> cpu,
            action_space = get_actions(env),
            dist = Normal,
            γ = 0.99f0,
            rng = rng,
        ),
        trajectory = ElasticCompactSARTSATrajectory(
            state_type = Float32,
            state_size = (ns,),
            action_type = Float32,
        ),
    )

    total_reward_per_episode = TotalRewardPerEpisode()
    time_per_step = TimePerStep()
    hook = ComposedHook(
        total_reward_per_episode,
        time_per_step,
        DoEveryNStep(10) do t, agent, env
            # log the distribution of action values.
            a = agent(env)
            with_logger(lg) do
                @info "step" action = a inner = TransformAction(a)
            end
        end,
        DoEveryNEpisode() do t, agent, env
            with_logger(lg) do
                @info(
                    "training",
                    loss = agent.policy.loss,
                    baseline_loss = agent.policy.baseline_loss,
                    reward = total_reward_per_episode.rewards[end],
                    log_step_increment = 0,
                )
            end
        end,
    )

    Experiment(
        agent,
        env,
        StopAfterEpisode(150),
        hook,
        Description("# Play Pendulum with VPG", save_dir),
    )
end
