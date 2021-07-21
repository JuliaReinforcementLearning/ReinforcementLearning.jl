function RL.Experiment(
    ::Val{:rlpyt},
    ::Val{:A2C},
    ::Val{:Atari},
    name::AbstractString;
    save_dir = nothing,
    seed = 123,
)
    @warn "Currently setting the `seed` will not guarantee the reproducibility. The instability seems to be caused by the `CrossCor` layer when calculating gradient."
    rng = StableRNG(seed)
    if isnothing(save_dir)
        t = Dates.format(now(), "yyyy_mm_dd_HH_MM_SS")
        save_dir = joinpath(pwd(), "checkpoints", "rlpyt_A2C_Atari_$(name)_$(t)")
    end

    lg = TBLogger(joinpath(save_dir, "tb_log"), min_level = Logging.Info)

    N_ENV = 32
    UPDATE_FREQ = 5
    N_FRAMES = 4
    STATE_SIZE = (80, 104)
    env = atari_env_factory(
        name,
        STATE_SIZE,
        N_FRAMES;
        repeat_action_probability = 0,
        seed = seed,
        n_replica = N_ENV,
    )
    N_ACTIONS = length(action_space(env[1]))

    init = orthogonal(rng)

    ## share model
    model = Chain(
        x -> x ./ 255,
        CrossCor((8, 8), N_FRAMES => 32, relu; stride = 4, pad = 0, init = init),
        CrossCor((4, 4), 32 => 64, relu; stride = 2, pad = 1, init = init),
        x -> reshape(x, :, size(x)[end]),
        Dense(6912, 512, relu; init = init),
    )

    agent = Agent(
        policy = RandomStartPolicy(
            num_rand_start = 1000,
            random_policy = RandomPolicy(action_space(env); rng = rng),
            policy = QBasedPolicy(
                learner = A2CLearner(
                    approximator = ActorCritic(
                        actor = Chain(model, Dense(512, N_ACTIONS; init = init)),
                        critic = Chain(model, Dense(512, 1; init = init)),
                        optimizer = ADAM(3e-4),
                    ) |> gpu,
                    γ = 0.99f0,
                    max_grad_norm = 1.0f0,
                    actor_loss_weight = 1.0f0,
                    critic_loss_weight = 0.25f0,
                    entropy_loss_weight = 0.01f0,
                    update_freq = UPDATE_FREQ,
                ),
                explorer = BatchExplorer(GumbelSoftmaxExplorer(; rng = rng)),
            ),
        ),
        trajectory = CircularArraySARTTrajectory(;
            capacity = UPDATE_FREQ,
            state = Array{Float32,4} => (STATE_SIZE..., N_FRAMES, N_ENV),
            action = Vector{Int} => (N_ENV,),
            reward = Vector{Float32} => (N_ENV,),
            terminal = Vector{Bool} => (N_ENV,),
        ),
    )

    N_TRAINING_STEPS = 50_000_000 ÷ N_ENV
    EVALUATION_FREQ = N_TRAINING_STEPS ÷ 100
    MAX_EPISODE_STEPS_EVAL = 27_000
    N_CHECKPOINTS = 3
    stop_condition = StopAfterStep(N_TRAINING_STEPS)

    total_batch_reward_per_episode = TotalBatchOriginalRewardPerEpisode(N_ENV)
    batch_steps_per_episode = BatchStepsPerEpisode(N_ENV)
    evaluation_result = []

    hook = ComposedHook(
        total_batch_reward_per_episode,
        batch_steps_per_episode,
        DoEveryNStep(;n=UPDATE_FREQ) do t, agent, env
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
                    total_batch_reward_per_episode.rewards[i][end] for i in 1:length(env) if is_terminated(env[i])
                ]
                if length(rewards) > 0
                    @info "training" rewards = mean(rewards) log_step_increment = 0
                end
                steps = [
                    batch_steps_per_episode.steps[i][end] for i in 1:length(env) if is_terminated(env[i])
                ]
                if length(steps) > 0
                    @info "training" steps = mean(steps) log_step_increment = 0
                end
            end
        end,
        DoEveryNStep(;n=EVALUATION_FREQ) do t, agent, env
            @info "evaluating agent at $t step..."
            h = TotalBatchOriginalRewardPerEpisode(N_ENV)
            s = @elapsed run(
                agent.policy,
                atari_env_factory(
                    name,
                    STATE_SIZE,
                    N_FRAMES,
                    MAX_EPISODE_STEPS_EVAL;
                    repeat_action_probability = 0,
                    seed = seed + t,
                    n_replica = 4,
                ),
                StopAfterStep(27_000; is_show_progress = false),
                h,
            )
            res = (avg_score = mean(Iterators.flatten(h.rewards)),)
            push!(evaluation_result, res)

            @info "finished evaluating agent in $s seconds" avg_score = res.avg_score
            with_logger(lg) do
                @info "evaluating" avg_score = res.avg_score log_step_increment = 0
            end

            policy = cpu(agent.policy)
            mkdir(joinpath(save_dir, string(t)))
            BSON.@save joinpath(save_dir, string(t), "policy.bson") policy
            BSON.@save joinpath(save_dir, string(t), "stats.bson") total_batch_reward_per_episode evaluation_result

            ## only keep recent 3 checkpoints
            old_checkpoint_folder =
                joinpath(save_dir, string(t - EVALUATION_FREQ * N_CHECKPOINTS))
            if isdir(old_checkpoint_folder)
                rm(old_checkpoint_folder; force = true, recursive = true)
            end
        end,
    )

    Experiment(agent, env, stop_condition, hook, "")
end
