function RL.Experiment(
    ::Val{:rlpyt},
    ::Val{:PPO},
    ::Val{:Atari},
    name::AbstractString;
    save_dir = nothing,
    seed = 123,
)
    @warn "Currently setting the `seed` will not guarantee the reproducibility. The instability seems to be caused by the `CrossCor` layer when calculating gradient."
    rng = StableRNG(seed)
    if isnothing(save_dir)
        t = Dates.format(now(), "yyyy_mm_dd_HH_MM_SS")
        save_dir = joinpath(pwd(), "checkpoints", "rlpyt_PPO_Atari_$(name)_$(t)")
    end

    lg = TBLogger(joinpath(save_dir, "tb_log"), min_level = Logging.Info)

    N_ENV = 32
    UPDATE_FREQ = 64
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
    INIT_CLIP_RANGE = 0.1f0
    INIT_LEARNING_RATE = 1e-3

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
        policy = PPOPolicy(
            approximator = ActorCritic(
                actor = Chain(model, Dense(512, N_ACTIONS; init = init)),
                critic = Chain(model, Dense(512, 1; init = init)),
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
            update_freq = UPDATE_FREQ,
            n_random_start = 1000,
        ),
        trajectory = PPOTrajectory(;
            capacity = UPDATE_FREQ,
            state = Array{Float32,4} => (STATE_SIZE..., N_FRAMES, N_ENV),
            action = Vector{Int} => (N_ENV,),
            reward = Vector{Float32} => (N_ENV,),
            terminal = Vector{Bool} => (N_ENV,),
            action_log_prob = Vector{Float32} => (N_ENV,),
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
        DoEveryNStep(; n = UPDATE_FREQ) do t, agent, env
            p = agent.policy
            with_logger(lg) do
                @info "training" loss = mean(p.loss) actor_loss = mean(p.actor_loss) critic_loss =
                    mean(p.critic_loss) entropy_loss = mean(p.entropy_loss) norm =
                    mean(p.norm) log_step_increment = UPDATE_FREQ
            end
        end,
        DoEveryNStep(; n = UPDATE_FREQ) do t, agent, env
            decay = (N_TRAINING_STEPS - t) / N_TRAINING_STEPS
            agent.policy.approximator.optimizer.eta = INIT_LEARNING_RATE * decay
            agent.policy.clip_range = INIT_CLIP_RANGE * Float32(decay)
        end,
        DoEveryNStep() do t, agent, env
            with_logger(lg) do
                rewards = [
                    total_batch_reward_per_episode.rewards[i][end] for
                    i in 1:length(env) if is_terminated(env[i])
                ]
                if length(rewards) > 0
                    @info "training" rewards = mean(rewards) log_step_increment = 0
                end
                steps = [
                    batch_steps_per_episode.steps[i][end] for
                    i in 1:length(env) if is_terminated(env[i])
                ]
                if length(steps) > 0
                    @info "training" steps = mean(steps) log_step_increment = 0
                end
            end
        end,
        DoEveryNStep(; n = EVALUATION_FREQ) do t, agent, env
            @info "evaluating agent at $t step..."
            ## switch to GreedyExplorer?
            h = TotalBatchRewardPerEpisode(N_ENV)
            s = @elapsed run(
                agent.policy,
                atari_env_factory(
                    name,
                    STATE_SIZE,
                    N_FRAMES,
                    MAX_EPISODE_STEPS_EVAL;
                    repeat_action_probability = 0,
                    seed = seed,
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
