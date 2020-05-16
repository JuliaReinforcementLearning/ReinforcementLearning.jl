export Experiment

using ArcadeLearningEnvironment
using Dates
using ReinforcementLearningCore
using ReinforcementLearningEnvironments
using Flux
using BSON
using TensorBoardLogger
using Logging
using Statistics

function RLCore.Experiment(
    ::Val{:Dopamine},
    ::Val{:DQN},
    ::Val{:Atari},
    name::AbstractString;
    save_dir = nothing,
)
    if isnothing(save_dir)
        t = Dates.format(now(), "yyyymmddHHMMSS")
        save_dir = joinpath(pwd(), "checkpoints", "dopamine_DQN_atari_$(name)_$(t)")
    end

    lg = TBLogger(joinpath(save_dir, "tb_log"), min_level = Logging.Info)

    N_FRAMES = 4
    STATE_SIZE = (84, 84)

    env_factory =
        () -> WrappedEnv(
            env = AtariEnv(;
                name = string(name),
                grayscale_obs = true,
                noop_max = 30,
                frame_skip = 4,
                terminal_on_life_loss = false,
                repeat_action_probability = 0.25,
                max_num_frames_per_episode = (name == "space_invaders" ? 3 : 4) * 100_000,  # https://github.com/openai/gym/blob/c33cfd8b2cc8cac6c346bc2182cd568ef33b8821/gym/envs/__init__.py#L621-L624
                color_averaging = false,
                full_action_space = false,
                #= seed=(22, 33) =#
            ),
            preprocessor = ComposedPreprocessor(
                ResizeImage(STATE_SIZE...),  # this implementation is different from cv2.resize https://github.com/google/dopamine/blob/e7d780d7c80954b7c396d984325002d60557f7d1/dopamine/discrete_domains/atari_lib.py#L629
                StackFrames(STATE_SIZE..., N_FRAMES),
            ),
        )

    env = env_factory()
    N_ACTIONS = length(get_action_space(env))

    init = seed_glorot_uniform()#= seed=341 =#

    create_model() =
        Chain(
            x -> x ./ 255,
            CrossCor((8, 8), N_FRAMES => 32, relu; stride = 4, pad = 2, init = init),
            CrossCor((4, 4), 32 => 64, relu; stride = 2, pad = 2, init = init),
            CrossCor((3, 3), 64 => 64, relu; stride = 1, pad = 1, init = init),
            x -> reshape(x, :, size(x)[end]),
            Dense(11 * 11 * 64, 512, relu),
            Dense(512, N_ACTIONS),
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
            ),
            explorer = EpsilonGreedyExplorer(
                ϵ_init = 1.0,
                ϵ_stable = 0.01,
                decay_steps = 250_000,
                kind = :linear,
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
    N_CHECKPOINTS = 3

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
                @info "training" reward = total_reward_per_episode.rewards[end] log_step_increment =
                    0
            end
        end,
        DoEveryNStep(EVALUATION_FREQ) do t, agent, env, obs
            @info "evaluating agent at $t step..."
            flush(stdout)
            old_explorer = agent.policy.explorer
            agent.policy.explorer = EpsilonGreedyExplorer(0.001)  # set evaluation epsilon
            Flux.testmode!(agent)
            h = ComposedHook(TotalRewardPerEpisode(), StepsPerEpisode())
            s = @elapsed run(
                agent,
                env_factory(),
                StopAfterStep(125_000; is_show_progress = false),
                h,
            )
            res = (
                avg_length = mean(h[2].steps[1:end-1]),
                avg_score = mean(h[1].rewards[1:end-1]),
            )
            push!(evaluation_result, res)
            Flux.trainmode!(agent)
            agent.policy.explorer = old_explorer
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
    This experiment uses alomost the same config in [dopamine](https://github.com/google/dopamine/blob/master/dopamine/agents/dqn/configs/dqn.gin). But do notice that there are some minor differences:

    - The RMSProp in Flux do not support center option (also the epsilon is not the same).
    - The image resize method used here is provided by ImageTransformers, which is not the same with the one in cv2.
    - `max_steps_per_episode` is not set, this might affect the evaluation result slightly.

    The testing environment is $name.
    Agent and statistic info will be saved to: $(joinpath(save_dir, string(N_TRAINING_STEPS)))
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
)
    if isnothing(save_dir)
        t = Dates.format(now(), "yyyymmddHHMMSS")
        save_dir = joinpath(pwd(), "checkpoints", "Dopamine_Dopamine_Atari_$(name)_$(t)")
    end

    lg = TBLogger(joinpath(save_dir, "tb_log"), min_level = Logging.Info)

    N_FRAMES = 4
    STATE_SIZE = (84, 84)

    env_factory =
        () -> WrappedEnv(
            env = AtariEnv(;
                name = string(name),
                grayscale_obs = true,
                noop_max = 30,
                frame_skip = 4,
                terminal_on_life_loss = false,
                repeat_action_probability = 0.25,
                max_num_frames_per_episode = (name == "space_invaders" ? 3 : 4) * 100_000,  # https://github.com/openai/gym/blob/c33cfd8b2cc8cac6c346bc2182cd568ef33b8821/gym/envs/__init__.py#L621-L624
                color_averaging = false,
                full_action_space = false,
                #= seed=(22, 33) =#
            ),
            preprocessor = ComposedPreprocessor(
                ResizeImage(STATE_SIZE...),  # this implementation is different from cv2.resize https://github.com/google/dopamine/blob/e7d780d7c80954b7c396d984325002d60557f7d1/dopamine/discrete_domains/atari_lib.py#L629
                StackFrames(STATE_SIZE..., N_FRAMES),
            ),
        )

    env = env_factory()
    N_ACTIONS = length(get_action_space(env))
    N_ATOMS = 51

    init = seed_glorot_uniform()#= seed=341 =#

    create_model() =
        Chain(
            x -> x ./ 255,
            CrossCor((8, 8), N_FRAMES => 32, relu; stride = 4, pad = 2, init = init),
            CrossCor((4, 4), 32 => 64, relu; stride = 2, pad = 2, init = init),
            CrossCor((3, 3), 64 => 64, relu; stride = 1, pad = 1, init = init),
            x -> reshape(x, :, size(x)[end]),
            Dense(11 * 11 * 64, 512, relu),
            Dense(512, N_ATOMS * N_ACTIONS),
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
            ),
            explorer = EpsilonGreedyExplorer(
                ϵ_init = 1.0,
                ϵ_stable = 0.01,
                decay_steps = 250_000,
                kind = :linear,
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
    N_CHECKPOINTS = 3

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
                @info "training" reward = total_reward_per_episode.rewards[end] log_step_increment =
                    0
            end
        end,
        DoEveryNStep(EVALUATION_FREQ) do t, agent, env, obs
            @info "evaluating agent at $t step..."
            flush(stdout)
            old_explorer = agent.policy.explorer
            agent.policy.explorer = EpsilonGreedyExplorer(0.001)  # set evaluation epsilon
            Flux.testmode!(agent)
            h = ComposedHook(TotalRewardPerEpisode(), StepsPerEpisode())
            s = @elapsed run(
                agent,
                env_factory(),
                StopAfterStep(125_000; is_show_progress = false),
                h,
            )
            res = (
                avg_length = mean(h[2].steps[1:end-1]),
                avg_score = mean(h[1].rewards[1:end-1]),
            )
            push!(evaluation_result, res)
            Flux.trainmode!(agent)
            agent.policy.explorer = old_explorer
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
    - `max_steps_per_episode` is not set, this might affect the evaluation result slightly.

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
