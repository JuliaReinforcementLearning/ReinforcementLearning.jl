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
    rng = StableRNG(seed)
    N_ENV = 16
    UPDATE_FREQ = 10
    env = MultiThreadEnv([
        CartPoleEnv(; T = Float32, rng = StableRNG(hash(seed + i))) for i in 1:N_ENV
    ])
    ns, na = length(state(env[1])), length(action_space(env[1]))
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
                update_freq = UPDATE_FREQ,
            ),
            explorer = BatchExplorer(GumbelSoftmaxExplorer(;)),
        ),
        trajectory = CircularArraySARTTrajectory(;
            capacity = UPDATE_FREQ,
            state = Matrix{Float32} => (ns, N_ENV),
            action = Vector{Int} => (N_ENV,),
            reward = Vector{Float32} => (N_ENV,),
            terminal = Vector{Bool} => (N_ENV,),
        ),
    )
    stop_condition = StopAfterStep(50_000)
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
                    if is_terminated(env[i])
                        @info "training" reward = total_reward_per_episode.rewards[i][end] log_step_increment =
                            0
                        break
                    end
                end
            end
        end,
    )
    Experiment(agent, env, stop_condition, hook, "# A2CGAE with CartPole")
end
