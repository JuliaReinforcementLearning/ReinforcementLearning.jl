export Experiment

using Dates
using ReinforcementLearningBase
using ReinforcementLearningCore
using ReinforcementLearningEnvironments
using Flux
using BSON
using TensorBoardLogger
using Logging

function RLCore.Experiment(::Val{:JuliaRL}, ::Val{:BasicDQN}, ::Val{:CartPole}, ::Nothing; save_dir=nothing)
    if isnothing(save_dir)
        t = Dates.format(now(), "yyyymmddHHMMSS")
        save_dir = joinpath(pwd(), "checkpoints", "juliarl_BasicDQN_CartPole_$(t)")
    end

    lg=TBLogger(joinpath(save_dir, "tb_log"), min_level=Logging.Info)

    env = CartPoleEnv(; T = Float32  , seed = 11  )
    ns, na = length(rand(get_observation_space(env))), length(get_action_space(env))
    agent = Agent(
        policy = QBasedPolicy(
            learner = BasicDQNLearner(
                approximator = NeuralNetworkApproximator(
                    model = Chain(
                        Dense(ns, 128, relu; initW = seed_glorot_uniform( seed = 17 )),
                        Dense(128, 128, relu; initW = seed_glorot_uniform( seed = 23 )),
                        Dense(128, na; initW = seed_glorot_uniform( seed = 39 )),
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
                @info "training" loss=agent.policy.learner.loss
            end
        end,
        DoEveryNStep(10000) do t, agent, env, obs
            RLCore.save(save_dir, agent)
            BSON.@save joinpath(save_dir, "stats.bson") total_reward_per_episode time_per_step
        end
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

function RLCore.Experiment(::Val{:JuliaRL}, ::Val{:DQN}, ::Val{:CartPole}, ::Nothing; save_dir=nothing)
    if isnothing(save_dir)
        t = Dates.format(now(), "yyyymmddHHMMSS")
        save_dir = joinpath(pwd(), "checkpoints", "juliarl_BasicDQN_CartPole_$(t)")
    end

    lg=TBLogger(joinpath(save_dir, "tb_log"), min_level=Logging.Info)

    env = CartPoleEnv(; T = Float32  , seed = 11  )
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
                    @info "training" loss=agent.policy.learner.loss
                end
            end
        end,
        DoEveryNStep(10000) do t, agent, env, obs
            RLCore.save(save_dir, agent)
            BSON.@save joinpath(save_dir, "stats.bson") total_reward_per_episode time_per_step
        end
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

function RLCore.Experiment(::Val{:JuliaRL}, ::Val{:PrioritizedDQN}, ::Val{:CartPole}, ::Nothing; save_dir=nothing)
    if isnothing(save_dir)
        t = Dates.format(now(), "yyyymmddHHMMSS")
        save_dir = joinpath(pwd(), "checkpoints", "juliarl_BasicDQN_CartPole_$(t)")
    end

    lg=TBLogger(joinpath(save_dir, "tb_log"), min_level=Logging.Info)

    env = CartPoleEnv(; T = Float32  , seed = 11  )
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
                    @info "training" loss=agent.policy.learner.loss
                end
            end
        end,
        DoEveryNStep(10000) do t, agent, env, obs
            RLCore.save(save_dir, agent)
            BSON.@save joinpath(save_dir, "stats.bson") total_reward_per_episode time_per_step
        end
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

function RLCore.Experiment(::Val{:JuliaRL}, ::Val{:Rainbow}, ::Val{:CartPole}, ::Nothing; save_dir=nothing)
    if isnothing(save_dir)
        t = Dates.format(now(), "yyyymmddHHMMSS")
        save_dir = joinpath(pwd(), "checkpoints", "juliarl_BasicDQN_CartPole_$(t)")
    end

    lg=TBLogger(joinpath(save_dir, "tb_log"), min_level=Logging.Info)

    env = CartPoleEnv(; T = Float32  , seed = 11  )
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
                    @info "training" loss=agent.policy.learner.loss
                end
            end
        end,
        DoEveryNStep(10000) do t, agent, env, obs
            RLCore.save(save_dir, agent)
            BSON.@save joinpath(save_dir, "stats.bson") total_reward_per_episode time_per_step
        end
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

function RLCore.Experiment(::Val{:JuliaRL}, ::Val{:A2C}, ::Val{:CartPole}, ::Nothing; save_dir=nothing)
    N_ENV = 16
    UPDATE_FREQ = 10
    env = MultiThreadEnv([CartPoleEnv(; T = Float32, seed = i) for i in 1:N_ENV])
    ns, na = length(rand(get_observation_space(env[1]))), length(get_action_space(env[1]))
    RLBase.reset!(env, is_force = true)
    agent = Agent(
        policy = QBasedPolicy(
            learner = A2CLearner(
                approximator = ActorCritic(
                    actor = NeuralNetworkApproximator(
                        model = Chain(
                            Dense(ns, 256, relu; initW = seed_glorot_uniform(seed = 17)),
                            Dense(256, na; initW = seed_glorot_uniform(seed = 23)),
                            softmax,
                        ),
                        optimizer = ADAM(1e-3)
                    ),
                    critic = NeuralNetworkApproximator(
                        model = Chain(
                            Dense(ns, 256, relu; initW = seed_glorot_uniform(seed = 29)),
                            Dense(256, 1; initW = seed_glorot_uniform(seed = 29)),
                        ),
                        optimizer = ADAM(1e-3)
                    ),
                ) |> cpu,
                γ = 0.99f0,
                actor_loss_weight = 1.0f0,
                critic_loss_weight = 0.5f0,
                entropy_loss_weight = 0.001f0,
            ),
            explorer = BatchExplorer((
                WeightedExplorer(; is_normalized = true, seed = s) for s in 10:9+N_ENV
            )...),
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
    );


    hook = TotalBatchRewardPerEpisode(N_ENV)
    stop_condition = StopAfterStep(100000)
    Experiment(agent, env, stop_condition, hook, "# A2C with CartPole")
end

function RLCore.Experiment(::Val{:JuliaRL}, ::Val{:A2CGAE}, ::Val{:CartPole}, ::Nothing; save_dir=nothing)
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
                            softmax,
                        ),
                        optimizer = ADAM(1e-3)
                    ),
                    critic = NeuralNetworkApproximator(
                        model = Chain(
                            Dense(ns, 256, relu; initW = seed_glorot_uniform(seed = 29)),
                            Dense(256, 1; initW = seed_glorot_uniform(seed = 29)),
                        ),
                        optimizer = ADAM(1e-3)
                    ),
                ) |> cpu,
                γ = 0.99f0,
                λ = 0.97f0,
                actor_loss_weight = 1.0f0,
                critic_loss_weight = 0.5f0,
                entropy_loss_weight = 0.001f0,
            ),
            explorer = BatchExplorer((
                WeightedExplorer(; is_normalized = true, seed = s) for s in 10:9+N_ENV
            )...),
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
    Experiment(agent, env, StopAfterStep(100000), TotalBatchRewardPerEpisode(N_ENV), "# A2CGAE with CartPole")
end
