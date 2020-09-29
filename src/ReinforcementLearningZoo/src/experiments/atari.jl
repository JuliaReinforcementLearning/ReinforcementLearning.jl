export Experiment

using .ArcadeLearningEnvironment
using Dates
using ReinforcementLearningCore
using .ReinforcementLearningEnvironments
using Flux
using BSON
using TensorBoardLogger
using Logging
using Statistics
using Random

function atari_env_factory(
    name,
    state_size,
    n_frames,
    max_episode_steps = 100_000;
    seed = nothing,
    repeat_action_probability = 0.25,
)
    AtariEnv(;
        name = string(name),
        grayscale_obs = true,
        noop_max = 30,
        frame_skip = 4,
        terminal_on_life_loss = false,
        repeat_action_probability = repeat_action_probability,
        max_num_frames_per_episode = n_frames * max_episode_steps,
        color_averaging = false,
        full_action_space = false,
        seed = seed,
    ) |>
    StateOverriddenEnv(
        ResizeImage(state_size...),  # this implementation is different from cv2.resize https://github.com/google/dopamine/blob/e7d780d7c80954b7c396d984325002d60557f7d1/dopamine/discrete_domains/atari_lib.py#L629
        StackFrames(state_size..., n_frames),
    ) |>
    StateCachedEnv |>
    RewardOverriddenEnv(r -> clamp(r, -1, 1))
end


function RLCore.Experiment(
    ::Val{:Dopamine},
    ::Val{:DQN},
    ::Val{:Atari},
    name::AbstractString;
    save_dir = nothing,
    seed = 123,
)

    @warn "Currently setting the `seed` will not guarantee the reproducibility. The instability seems to be caused by the `CrossCor` layer when calculating gradient."
    rng = MersenneTwister(seed)
    if isnothing(save_dir)
        t = Dates.format(now(), "yyyy_mm_dd_HH_MM_SS")
        save_dir = joinpath(pwd(), "checkpoints", "dopamine_DQN_atari_$(name)_$(t)")
    end

    lg = TBLogger(joinpath(save_dir, "tb_log"), min_level = Logging.Info)

    N_FRAMES = 4
    STATE_SIZE = (84, 84)
    env = atari_env_factory(name, STATE_SIZE, N_FRAMES; seed = hash(seed + 1))
    N_ACTIONS = length(get_actions(env))
    init = glorot_uniform(rng)

    create_model() =
        Chain(
            x -> x ./ 255,
            CrossCor((8, 8), N_FRAMES => 32, relu; stride = 4, pad = 2, init = init),
            CrossCor((4, 4), 32 => 64, relu; stride = 2, pad = 2, init = init),
            CrossCor((3, 3), 64 => 64, relu; stride = 1, pad = 1, init = init),
            x -> reshape(x, :, size(x)[end]),
            Dense(11 * 11 * 64, 512, relu; initW = init),
            Dense(512, N_ACTIONS; initW = init),
        ) |> gpu

    agent = Agent(
        policy = QBasedPolicy(
            learner = DQNLearner(
                approximator = NeuralNetworkApproximator(
                    model = create_model(),
                    optimizer = RMSProp(0.00025, 0.95),
                ),  # unlike TF/PyTorch RMSProp doesn't support center
                target_approximator = NeuralNetworkApproximator(model = create_model()),
                update_freq = 4,
                γ = 0.99f0,
                update_horizon = 1,
                batch_size = 32,
                stack_size = N_FRAMES,
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
        trajectory = CircularCompactSARTSATrajectory(
            capacity = 1_000_000,
            state_type = Float32,
            state_size = STATE_SIZE,
        ),
    )

    evaluation_result = []
    EVALUATION_FREQ = 250_000
    MAX_EPISODE_STEPS_EVAL = 27_000
    N_CHECKPOINTS = 3

    total_reward_per_episode = TotalRewardPerEpisode()
    time_per_step = TimePerStep()
    steps_per_episode = StepsPerEpisode()
    hook = ComposedHook(
        total_reward_per_episode,
        time_per_step,
        steps_per_episode,
        DoEveryNStep() do t, agent, env
            with_logger(lg) do
                @info "training" loss = agent.policy.learner.loss
            end
        end,
        DoEveryNEpisode() do t, agent, env
            with_logger(lg) do
                @info "training" episode_length = steps_per_episode.steps[end] reward =
                    total_reward_per_episode.rewards[end] log_step_increment = 0
            end
        end,
        DoEveryNStep(EVALUATION_FREQ) do t, agent, env
            @info "evaluating agent at $t step..."
            flush(stdout)
            Flux.testmode!(agent)
            old_explorer = agent.policy.explorer
            agent.policy.explorer = EpsilonGreedyExplorer(0.001; rng = rng)  # set evaluation epsilon
            h = ComposedHook(TotalRewardPerEpisode(), StepsPerEpisode())
            s = @elapsed run(
                agent,
                atari_env_factory(
                    name,
                    STATE_SIZE,
                    N_FRAMES,
                    MAX_EPISODE_STEPS_EVAL;
                    seed = hash(seed + t),
                ),
                StopAfterStep(125_000; is_show_progress = false),
                h,
            )
            res = (
                avg_length = mean(h[2].steps[1:end-1]),
                avg_score = mean(h[1].rewards[1:end-1]),
            )
            push!(evaluation_result, res)
            agent.policy.explorer = old_explorer
            Flux.trainmode!(agent)
            @info "finished evaluating agent in $s seconds" avg_length = res.avg_length avg_score =
                res.avg_score
            with_logger(lg) do
                @info "evaluating" avg_length = res.avg_length avg_score = res.avg_score log_step_increment = 0
            end
            flush(stdout)

            RLCore.save(joinpath(save_dir, string(t)), agent; is_save_trajectory = false)  # saving trajectory will take about 27G disk space each time
            BSON.@save joinpath(save_dir, string(t), "stats.bson") total_reward_per_episode time_per_step evaluation_result

            # only keep recent 3 checkpoints
            old_checkpoint_folder =
                joinpath(save_dir, string(t - EVALUATION_FREQ * N_CHECKPOINTS))
            if isdir(old_checkpoint_folder)
                rm(old_checkpoint_folder; force = true, recursive = true)
            end
        end,
    )

    N_TRAINING_STEPS = 50_000_000
    stop_condition = StopAfterStep(N_TRAINING_STEPS)

    description = """
    This experiment uses almost the same config in [dopamine](https://github.com/google/dopamine/blob/master/dopamine/agents/dqn/configs/dqn.gin). But do notice that there are some minor differences:

    - The RMSProp in Flux do not support center option (also the epsilon is not the same).
    - The image resize method used here is provided by ImageTransformers, which is not the same with the one in cv2.

    The testing environment is $name.
    Agent and statistic info will be saved to: `$(joinpath(save_dir, string(N_TRAINING_STEPS)))`
    You can also view the tensorboard logs with `tensorboard --logdir $(joinpath(save_dir, "tb_log"))`

    To load the agent and statistic info:
    ```
    agent = RLCore.load("$(joinpath(save_dir, string(N_TRAINING_STEPS)))", Agent)
    BSON.@load joinpath("$(joinpath(save_dir, string(N_TRAINING_STEPS)))", "stats.bson") total_reward_per_episode time_per_step evaluation_result
    ```
    """

    Experiment(agent, env, stop_condition, hook, description)
end

function RLCore.Experiment(
    ::Val{:Dopamine},
    ::Val{:Rainbow},
    ::Val{:Atari},
    name::AbstractString;
    save_dir = nothing,
    seed = 123,
)
    @warn "Currently setting the `seed` will not guarantee the reproducibility. The instability seems to be caused by the `CrossCor` layer when calculating gradient."
    rng = MersenneTwister(seed)
    if isnothing(save_dir)
        t = Dates.format(now(), "yyyy_mm_dd_HH_MM_SS")
        save_dir = joinpath(pwd(), "checkpoints", "Dopamine_Rainbow_Atari_$(name)_$(t)")
    end

    lg = TBLogger(joinpath(save_dir, "tb_log"), min_level = Logging.Info)

    N_FRAMES = 4
    STATE_SIZE = (84, 84)
    env = atari_env_factory(name, STATE_SIZE, N_FRAMES; seed = hash(seed + 1))
    N_ACTIONS = length(get_actions(env))
    N_ATOMS = 51
    init = glorot_uniform(rng)

    create_model() =
        Chain(
            x -> x ./ 255,
            CrossCor((8, 8), N_FRAMES => 32, relu; stride = 4, pad = 2, init = init),
            CrossCor((4, 4), 32 => 64, relu; stride = 2, pad = 2, init = init),
            CrossCor((3, 3), 64 => 64, relu; stride = 1, pad = 1, init = init),
            x -> reshape(x, :, size(x)[end]),
            Dense(11 * 11 * 64, 512, relu; initW = init),
            Dense(512, N_ATOMS * N_ACTIONS; initW = init),
        ) |> gpu

    agent = Agent(
        policy = QBasedPolicy(
            learner = RainbowLearner(
                approximator = NeuralNetworkApproximator(
                    model = create_model(),
                    optimizer = ADAM(0.0000625),
                ),  # epsilon is not set here
                target_approximator = NeuralNetworkApproximator(model = create_model()),
                n_actions = N_ACTIONS,
                n_atoms = N_ATOMS,
                Vₘₐₓ = 10.0f0,
                Vₘᵢₙ = -10.0f0,
                update_freq = 4,
                γ = 0.99f0,
                update_horizon = 3,
                batch_size = 32,
                stack_size = N_FRAMES,
                min_replay_history = 20_000,
                loss_func = logitcrossentropy_unreduced,
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
        trajectory = CircularCompactPSARTSATrajectory(
            capacity = 1_000_000,
            state_type = Float32,
            state_size = STATE_SIZE,
        ),
    )

    evaluation_result = []
    EVALUATION_FREQ = 250_000
    MAX_EPISODE_STEPS_EVAL = 27_000
    N_CHECKPOINTS = 3

    total_reward_per_episode = TotalRewardPerEpisode()
    time_per_step = TimePerStep()
    steps_per_episode = StepsPerEpisode()
    hook = ComposedHook(
        total_reward_per_episode,
        time_per_step,
        steps_per_episode,
        DoEveryNStep() do t, agent, env
            with_logger(lg) do
                @info "training" loss = agent.policy.learner.loss
            end
        end,
        DoEveryNEpisode() do t, agent, env
            with_logger(lg) do
                @info "training" reward = total_reward_per_episode.rewards[end] episode_length =
                    steps_per_episode.steps[end] log_step_increment = 0
            end
        end,
        DoEveryNStep(EVALUATION_FREQ) do t, agent, env
            @info "evaluating agent at $t step..."
            flush(stdout)
            Flux.testmode!(agent)
            old_explorer = agent.policy.explorer
            agent.policy.explorer = EpsilonGreedyExplorer(0.001; rng = rng)  # set evaluation epsilon
            h = ComposedHook(TotalRewardPerEpisode(), StepsPerEpisode())
            s = @elapsed run(
                agent,
                atari_env_factory(
                    name,
                    STATE_SIZE,
                    N_FRAMES,
                    MAX_EPISODE_STEPS_EVAL;
                    seed = hash(seed + t),
                ),
                StopAfterStep(125_000; is_show_progress = false),
                h,
            )
            res = (
                avg_length = mean(h[2].steps[1:end-1]),
                avg_score = mean(h[1].rewards[1:end-1]),
            )
            push!(evaluation_result, res)
            agent.policy.explorer = old_explorer
            Flux.trainmode!(agent)
            @info "finished evaluating agent in $s seconds" avg_length = res.avg_length avg_score =
                res.avg_score
            with_logger(lg) do
                @info "evaluating" avg_length = res.avg_length avg_score = res.avg_score log_step_increment = 0
            end
            flush(stdout)

            RLCore.save(joinpath(save_dir, string(t)), agent; is_save_trajectory = false)  # saving trajectory will take about 27G disk space each time
            BSON.@save joinpath(save_dir, string(t), "stats.bson") total_reward_per_episode time_per_step evaluation_result

            # only keep recent 3 checkpoints
            old_checkpoint_folder =
                joinpath(save_dir, string(t - EVALUATION_FREQ * N_CHECKPOINTS))
            if isdir(old_checkpoint_folder)
                rm(old_checkpoint_folder; force = true, recursive = true)
            end
        end,
    )

    N_TRAINING_STEPS = 50_000_000
    stop_condition = StopAfterStep(N_TRAINING_STEPS)

    description = """
    This experiment uses alomost the same config in [dopamine](https://github.com/google/dopamine/blob/master/dopamine/agents/rainbow/configs/rainbow.gin). But do notice that there are some minor differences:

    - The epsilon in ADAM optimizer is not changed
    - The image resize method used here is provided by ImageTransformers, which is not the same with the one in cv2.

    The testing environment is $name.
    Agent and statistic info will be saved to: `$(joinpath(save_dir, string(N_TRAINING_STEPS)))`
    You can also view the tensorboard logs with `tensorboard --logdir $(joinpath(save_dir, "tb_log"))`

    To load the agent and statistic info:
    ```
    agent = RLCore.load("$(joinpath(save_dir, string(N_TRAINING_STEPS)))", Agent)
    BSON.@load joinpath("$(joinpath(save_dir, string(N_TRAINING_STEPS)))", "stats.bson") total_reward_per_episode time_per_step evaluation_result
    ```
    """

    Experiment(agent, env, stop_condition, hook, description)
end

function RLCore.Experiment(
    ::Val{:Dopamine},
    ::Val{:IQN},
    ::Val{:Atari},
    name::AbstractString;
    save_dir = nothing,
    seed = 123,
)
    @warn "Currently setting the `seed` will not guarantee the reproducibility. The instability seems to be caused by the `CrossCor` layer when calculating gradient."
    rng = MersenneTwister(seed)
    device_rng = CUDA.CURAND.RNG()
    Random.seed!(device_rng, hash(seed + 1))
    if isnothing(save_dir)
        t = Dates.format(now(), "yyyy_mm_dd_HH_MM_SS")
        save_dir = joinpath(pwd(), "checkpoints", "Dopamine_IQN_Atari_$(name)_$(t)")
    end

    lg = TBLogger(joinpath(save_dir, "tb_log"), min_level = Logging.Info)

    N_FRAMES = 4
    STATE_SIZE = (84, 84)
    MAX_STEPS_PER_EPISODE = 27_000

    env = atari_env_factory(name, STATE_SIZE, N_FRAMES; seed = hash(seed + 2))
    N_ACTIONS = length(get_actions(env))
    Nₑₘ = 64

    init = glorot_uniform(rng)

    create_model() =
        ImplicitQuantileNet(
            ψ = Chain(
                x -> x ./ 255,
                CrossCor((8, 8), N_FRAMES => 32, relu; stride = 4, pad = 2, init = init),
                CrossCor((4, 4), 32 => 64, relu; stride = 2, pad = 2, init = init),
                CrossCor((3, 3), 64 => 64, relu; stride = 1, pad = 1, init = init),
                x -> reshape(x, :, size(x)[end]),
            ),
            ϕ = Dense(Nₑₘ, 11 * 11 * 64, relu; initW = init),
            header = Chain(
                Dense(11 * 11 * 64, 512, relu; initW = init),
                Dense(512, N_ACTIONS; initW = init),
            ),
        ) |> gpu

    agent = Agent(
        policy = QBasedPolicy(
            learner = IQNLearner(
                approximator = NeuralNetworkApproximator(
                    model = create_model(),
                    optimizer = ADAM(0.00005),  # epsilon is not set here
                ),
                target_approximator = NeuralNetworkApproximator(model = create_model()),
                κ = 1.0f0,
                N = 64,
                N′ = 64,
                Nₑₘ = Nₑₘ,
                K = 32,
                γ = 0.99f0,
                stack_size = 4,
                batch_size = 32,
                update_horizon = 3,
                min_replay_history = 20_000,
                update_freq = 4,
                target_update_freq = 8_000,
                default_priority = 1.0f2,
                rng = rng,
                device_rng = device_rng,
            ),
            explorer = EpsilonGreedyExplorer(
                ϵ_init = 1.0,
                ϵ_stable = 0.01,
                decay_steps = 250_000,
                kind = :linear,
                rng = rng,
            ),
        ),
        trajectory = CircularCompactSARTSATrajectory(
            capacity = 1_000_000,
            state_type = Float32,
            state_size = STATE_SIZE,
        ),
    )

    evaluation_result = []
    EVALUATION_FREQ = 250_000
    MAX_EPISODE_STEPS_EVAL = 27_000
    N_CHECKPOINTS = 3

    total_reward_per_episode = TotalRewardPerEpisode()
    time_per_step = TimePerStep()
    steps_per_episode = StepsPerEpisode()
    hook = ComposedHook(
        total_reward_per_episode,
        time_per_step,
        steps_per_episode,
        DoEveryNStep() do t, agent, env
            with_logger(lg) do
                @info "training" loss = agent.policy.learner.loss
            end
        end,
        DoEveryNEpisode() do t, agent, env
            with_logger(lg) do
                @info "training" reward = total_reward_per_episode.rewards[end] episode_length =
                    steps_per_episode.steps[end] log_step_increment = 0
            end
        end,
        DoEveryNStep(EVALUATION_FREQ) do t, agent, env
            @info "evaluating agent at $t step..."
            flush(stdout)
            Flux.testmode!(agent)
            old_explorer = agent.policy.explorer
            agent.policy.explorer = EpsilonGreedyExplorer(0.001; rng = rng)  # set evaluation epsilon
            h = ComposedHook(TotalRewardPerEpisode(), StepsPerEpisode())
            s = @elapsed run(
                agent,
                atari_env_factory(
                    name,
                    STATE_SIZE,
                    N_FRAMES,
                    MAX_EPISODE_STEPS_EVAL;
                    seed = hash(seed + t),
                ),
                StopAfterStep(125_000; is_show_progress = false),
                h,
            )
            res = (
                avg_length = mean(h[2].steps[1:end-1]),
                avg_score = mean(h[1].rewards[1:end-1]),
            )
            push!(evaluation_result, res)
            agent.policy.explorer = old_explorer
            Flux.trainmode!(agent)
            @info "finished evaluating agent in $s seconds" avg_length = res.avg_length avg_score =
                res.avg_score
            with_logger(lg) do
                @info "evaluating" avg_length = res.avg_length avg_score = res.avg_score log_step_increment = 0
            end
            flush(stdout)

            RLCore.save(joinpath(save_dir, string(t)), agent; is_save_trajectory = false)  # saving trajectory will take about 27G disk space each time
            BSON.@save joinpath(save_dir, string(t), "stats.bson") total_reward_per_episode time_per_step evaluation_result

            # only keep recent 3 checkpoints
            old_checkpoint_folder =
                joinpath(save_dir, string(t - EVALUATION_FREQ * N_CHECKPOINTS))
            if isdir(old_checkpoint_folder)
                rm(old_checkpoint_folder; force = true, recursive = true)
            end
        end,
    )

    N_TRAINING_STEPS = 50_000_000
    stop_condition = StopAfterStep(N_TRAINING_STEPS)

    description = """
    This experiment uses alomost the same config in [dopamine](https://github.com/google/dopamine/blob/master/dopamine/agents/implicit_quantile/configs/implicit_quantile.gin). But do notice that there are some minor differences:

    - The epsilon in ADAM optimizer is not changed
    - The image resize method used here is provided by ImageTransformers, which is not the same with the one in cv2.

    The testing environment is $name.
    Agent and statistic info will be saved to: `$(joinpath(save_dir, string(N_TRAINING_STEPS)))`
    You can also view the tensorboard logs with `tensorboard --logdir $(joinpath(save_dir, "tb_log"))`

    To load the agent and statistic info:
    ```
    agent = RLCore.load("$(joinpath(save_dir, string(N_TRAINING_STEPS)))", Agent)
    BSON.@load joinpath("$(joinpath(save_dir, string(N_TRAINING_STEPS)))", "stats.bson") total_reward_per_episode time_per_step evaluation_result
    ```
    """

    Experiment(agent, env, stop_condition, hook, description)
end

function RLCore.Experiment(
    ::Val{:rlpyt},
    ::Val{:A2C},
    ::Val{:Atari},
    name::AbstractString;
    save_dir = nothing,
    seed = 123,
)
    @warn "Currently setting the `seed` will not guarantee the reproducibility. The instability seems to be caused by the `CrossCor` layer when calculating gradient."
    rng = MersenneTwister(seed)
    if isnothing(save_dir)
        t = Dates.format(now(), "yyyy_mm_dd_HH_MM_SS")
        save_dir = joinpath(pwd(), "checkpoints", "rlpyt_A2C_Atari_$(name)_$(t)")
    end

    lg = TBLogger(joinpath(save_dir, "tb_log"), min_level = Logging.Info)

    N_ENV = 32
    UPDATE_FREQ = 5
    N_FRAMES = 4
    STATE_SIZE = (80, 104)
    env = MultiThreadEnv([
        atari_env_factory(
            name,
            STATE_SIZE,
            N_FRAMES;
            repeat_action_probability = 0,
            seed = hash(seed + i),
        ) for i in 1:N_ENV
    ])
    N_ACTIONS = length(get_actions(env[1]))

    init = orthogonal(rng)

    # share model
    model = Chain(
        x -> x ./ 255,
        CrossCor((8, 8), N_FRAMES => 32, relu; stride = 4, pad = 0, init = init),
        CrossCor((4, 4), 32 => 64, relu; stride = 2, pad = 1, init = init),
        x -> reshape(x, :, size(x)[end]),
        Dense(6912, 512, relu; initW = init),
    )

    agent = Agent(
        policy = RandomStartPolicy(
            num_rand_start = 1000,
            random_policy = RandomPolicy(get_actions(env); rng = rng),
            policy = QBasedPolicy(
                learner = A2CLearner(
                    approximator = ActorCritic(
                        actor = Chain(model, Dense(512, N_ACTIONS; initW = init)),
                        critic = Chain(model, Dense(512, 1; initW = init)),
                        optimizer = ADAM(3e-4),
                    ) |> gpu,
                    γ = 0.99f0,
                    max_grad_norm = 1.0f0,
                    actor_loss_weight = 1.0f0,
                    critic_loss_weight = 0.25f0,
                    entropy_loss_weight = 0.01f0,
                ),
                explorer = BatchExplorer(GumbelSoftmaxExplorer(; rng = rng)),
            ),
        ),
        trajectory = CircularCompactSARTSATrajectory(;
            capacity = UPDATE_FREQ,
            state_type = Float32,
            state_size = (STATE_SIZE..., N_FRAMES, N_ENV),
            action_type = Int,
            action_size = (N_ENV,),
            reward_type = Float32,
            reward_size = (N_ENV,),
            terminal_type = Bool,
            terminal_size = (N_ENV,),
        ),
    )

    N_TRAINING_STEPS = 50_000_000 ÷ N_ENV
    EVALUATION_FREQ = N_TRAINING_STEPS ÷ 100
    MAX_EPISODE_STEPS_EVAL = 27_000
    N_CHECKPOINTS = 3
    stop_condition = StopAfterStep(N_TRAINING_STEPS)

    total_batch_reward_per_episode = TotalBatchRewardPerEpisode(N_ENV)
    batch_steps_per_episode = BatchStepsPerEpisode(N_ENV)
    evaluation_result = []

    hook = ComposedHook(
        total_batch_reward_per_episode,
        batch_steps_per_episode,
        DoEveryNStep(UPDATE_FREQ) do t, agent, env
            learner = agent.policy.policy.learner
            with_logger(lg) do
                @info "training" loss = learner.loss actor_loss = learner.actor_loss critic_loss =
                    learner.critic_loss entropy_loss = learner.entropy_loss norm =
                    learner.norm log_step_increment = UPDATE_FREQ
            end
        end,
        DoEveryNStep() do t, agent, env
            with_logger(lg) do
                rewards = [
                    total_batch_reward_per_episode.rewards[i][end]
                    for i in 1:length(env) if get_terminal(env[i])
                ]
                if length(rewards) > 0
                    @info "training" rewards = mean(rewards) log_step_increment = 0
                end
                steps = [
                    batch_steps_per_episode.steps[i][end]
                    for i in 1:length(env) if get_terminal(env[i])
                ]
                if length(steps) > 0
                    @info "training" steps = mean(steps) log_step_increment = 0
                end
            end
        end,
        DoEveryNStep(EVALUATION_FREQ) do t, agent, env
            @info "evaluating agent at $t step..."
            flush(stdout)
            Flux.testmode!(agent)
            h = TotalBatchRewardPerEpisode(N_ENV)
            s = @elapsed run(
                agent,
                MultiThreadEnv([
                    atari_env_factory(
                        name,
                        STATE_SIZE,
                        N_FRAMES,
                        MAX_EPISODE_STEPS_EVAL;
                        repeat_action_probability = 0,
                        seed = hash(seed + t + i),
                    ) for i in 1:N_ENV
                ]),
                StopAfterStep(27_000; is_show_progress = false),
                h,
            )
            res = (avg_score = mean(Iterators.flatten(h.rewards)),)
            push!(evaluation_result, res)
            Flux.trainmode!(agent)
            @info "finished evaluating agent in $s seconds" avg_score = res.avg_score
            with_logger(lg) do
                @info "evaluating" avg_score = res.avg_score log_step_increment = 0
            end
            flush(stdout)

            RLCore.save(joinpath(save_dir, string(t)), agent;)
            BSON.@save joinpath(save_dir, string(t), "stats.bson") total_batch_reward_per_episode evaluation_result

            # only keep recent 3 checkpoints
            old_checkpoint_folder =
                joinpath(save_dir, string(t - EVALUATION_FREQ * N_CHECKPOINTS))
            if isdir(old_checkpoint_folder)
                rm(old_checkpoint_folder; force = true, recursive = true)
            end
        end,
    )

    description = """
    # Play Atari($name) with A2C

    The configurations use here should be aligned with [atari_ff_a2c.py](https://github.com/astooke/rlpyt/blob/master/rlpyt/experiments/configs/atari/pg/atari_ff_a2c.py) in rlpyt.

    You can also view the tensorboard logs with `tensorboard --logdir $(joinpath(save_dir, "tb_log"))`
    """

    Experiment(agent, env, stop_condition, hook, description)
end

function RLCore.Experiment(
    ::Val{:rlpyt},
    ::Val{:PPO},
    ::Val{:Atari},
    name::AbstractString;
    save_dir = nothing,
    seed = 123,
)
    @warn "Currently setting the `seed` will not guarantee the reproducibility. The instability seems to be caused by the `CrossCor` layer when calculating gradient."
    rng = MersenneTwister(seed)
    if isnothing(save_dir)
        t = Dates.format(now(), "yyyy_mm_dd_HH_MM_SS")
        save_dir = joinpath(pwd(), "checkpoints", "rlpyt_PPO_Atari_$(name)_$(t)")
    end

    lg = TBLogger(joinpath(save_dir, "tb_log"), min_level = Logging.Info)

    N_ENV = 32
    UPDATE_FREQ = 64
    N_FRAMES = 4
    STATE_SIZE = (80, 104)
    env = MultiThreadEnv([
        atari_env_factory(
            name,
            STATE_SIZE,
            N_FRAMES;
            repeat_action_probability = 0,
            seed = seed + i,
        ) for i in 1:N_ENV
    ])
    N_ACTIONS = length(get_actions(env[1]))
    INIT_CLIP_RANGE = 0.1f0
    INIT_LEARNING_RATE = 1e-3

    init = orthogonal(rng)

    # share model
    model = Chain(
        x -> x ./ 255,
        CrossCor((8, 8), N_FRAMES => 32, relu; stride = 4, pad = 0, init = init),
        CrossCor((4, 4), 32 => 64, relu; stride = 2, pad = 1, init = init),
        x -> reshape(x, :, size(x)[end]),
        Dense(6912, 512, relu; initW = init),
    )

    agent = Agent(
        policy = RandomStartPolicy(
            num_rand_start = 1000,
            random_policy = RandomPolicy(get_actions(env); rng = rng),
            policy = PPOPolicy(
                approximator = ActorCritic(
                    actor = Chain(model, Dense(512, N_ACTIONS; initW = init)),
                    critic = Chain(model, Dense(512, 1; initW = init)),
                    optimizer = ADAM(INIT_LEARNING_RATE),  # decrease learning rate with a hook
                ) |> gpu,
                γ = 0.99f0,
                λ = 0.98f0,
                clip_range = INIT_CLIP_RANGE,  # decrease with a hook
                max_grad_norm = 1.0f0,
                n_microbatches = 4,
                n_epochs = 4,
                actor_loss_weight = 1.0f0,
                critic_loss_weight = 0.5f0,
                entropy_loss_weight = 0.01f0,
                rng = rng,
            ),
        ),
        trajectory = PPOTrajectory(;
            capacity = UPDATE_FREQ,
            state_type = Float32,
            state_size = (STATE_SIZE..., N_FRAMES, N_ENV),
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

    N_TRAINING_STEPS = 50_000_000 ÷ N_ENV
    EVALUATION_FREQ = N_TRAINING_STEPS ÷ 100
    MAX_EPISODE_STEPS_EVAL = 27_000
    N_CHECKPOINTS = 3
    stop_condition = StopAfterStep(N_TRAINING_STEPS)

    total_batch_reward_per_episode = TotalBatchRewardPerEpisode(N_ENV)
    batch_steps_per_episode = BatchStepsPerEpisode(N_ENV)
    evaluation_result = []

    hook = ComposedHook(
        total_batch_reward_per_episode,
        batch_steps_per_episode,
        DoEveryNStep(UPDATE_FREQ) do t, agent, env
            p = agent.policy.policy
            with_logger(lg) do
                @info "training" loss = mean(p.loss) actor_loss =
                    mean(p.actor_loss) critic_loss = mean(p.critic_loss) entropy_loss =
                    mean(p.entropy_loss) norm = mean(p.norm) log_step_increment =
                    UPDATE_FREQ
            end
        end,
        DoEveryNStep(UPDATE_FREQ) do t, agent, env
            decay = (N_TRAINING_STEPS - t) / N_TRAINING_STEPS
            agent.policy.policy.approximator.optimizer.eta =
                INIT_LEARNING_RATE * decay
            agent.policy.policy.clip_range = INIT_CLIP_RANGE * Float32(decay)
        end,
        DoEveryNStep() do t, agent, env
            with_logger(lg) do
                rewards = [
                    total_batch_reward_per_episode.rewards[i][end]
                    for i in 1:length(env) if get_terminal(env[i])
                ]
                if length(rewards) > 0
                    @info "training" rewards = mean(rewards) log_step_increment = 0
                end
                steps = [
                    batch_steps_per_episode.steps[i][end]
                    for i in 1:length(env) if get_terminal(env[i])
                ]
                if length(steps) > 0
                    @info "training" steps = mean(steps) log_step_increment = 0
                end
            end
        end,
        DoEveryNStep(EVALUATION_FREQ) do t, agent, env
            @info "evaluating agent at $t step..."
            flush(stdout)
            Flux.testmode!(agent)
            # switch to GreedyExplorer?
            h = TotalBatchRewardPerEpisode(N_ENV)
            s = @elapsed run(
                agent,
                MultiThreadEnv([
                    atari_env_factory(
                        name,
                        STATE_SIZE,
                        N_FRAMES,
                        MAX_EPISODE_STEPS_EVAL;
                        repeat_action_probability = 0,
                        seed = seed + t + i,
                    ) for i in 1:4
                ]),
                StopAfterStep(27_000; is_show_progress = false),
                h,
            )
            res = (avg_score = mean(Iterators.flatten(h.rewards)),)
            push!(evaluation_result, res)
            Flux.trainmode!(agent)
            @info "finished evaluating agent in $s seconds" avg_score = res.avg_score
            with_logger(lg) do
                @info "evaluating" avg_score = res.avg_score log_step_increment = 0
            end
            flush(stdout)

            RLCore.save(joinpath(save_dir, string(t)), agent;)
            BSON.@save joinpath(save_dir, string(t), "stats.bson") total_batch_reward_per_episode evaluation_result

            # only keep recent 3 checkpoints
            old_checkpoint_folder =
                joinpath(save_dir, string(t - EVALUATION_FREQ * N_CHECKPOINTS))
            if isdir(old_checkpoint_folder)
                rm(old_checkpoint_folder; force = true, recursive = true)
            end
        end,
    )

    description = """
    # Play Atari($name) with PPO
    The configurations use here should be aligned with [atari_ff_ppo.py](https://github.com/astooke/rlpyt/blob/master/rlpyt/experiments/configs/atari/pg/atari_ff_ppo.py) in rlpyt.
    You can also view the tensorboard logs with `tensorboard --logdir $(joinpath(save_dir, "tb_log"))`
    """

    Experiment(agent, env, stop_condition, hook, description)
end
