function RLCore.Experiment(
    ::Val{:JuliaRL},
    ::Val{:MAC},
    ::Val{:CartPole},
    ::Nothing;
    save_dir = nothing,
    seed = 123,
)
    if isnothing(save_dir)
        t = Dates.format(now(), "yyyy_mm_dd_HH_MM_SS")
        save_dir = joinpath(pwd(), "checkpoints", "JuliaRL_MAC_CartPole_$(t)")
    end

    lg = TBLogger(joinpath(save_dir, "tb_log"), min_level = Logging.Info)
    rng = MersenneTwister(seed)
    N_ENV = 16
    UPDATE_FREQ = 20
    env = MultiThreadEnv([
        CartPoleEnv(; T = Float32, rng = MersenneTwister(hash(seed + i))) for i in 1:N_ENV
    ])
    ns, na = length(get_state(env[1])), length(get_actions(env[1]))
    RLBase.reset!(env, is_force = true)

    agent = Agent(
        policy = QBasedPolicy(
            learner = MACLearner(
                approximator = ActorCritic(
                    actor = NeuralNetworkApproximator(
                        model = Chain(
                            Dense(ns, 30, relu; initW = glorot_uniform(rng)),
                            Dense(30, 30, relu; initW = glorot_uniform(rng)),
                            Dense(30, na; initW = glorot_uniform(rng)),
                        ),
                        optimizer = ADAM(1e-2),
                    ),
                    critic = NeuralNetworkApproximator(
                        model = Chain(
                            Dense(ns, 30, relu; initW = glorot_uniform(rng)),
                            Dense(30, 30, relu; initW = glorot_uniform(rng)),
                            Dense(30, na; initW = glorot_uniform(rng)),
                        ),
                        optimizer = ADAM(3e-3),
                    ),
                ) |> cpu,
                γ = 0.99f0,
                bootstrap = true,
                update_freq = UPDATE_FREQ
            ),
            explorer = BatchExplorer(GumbelSoftmaxExplorer()),#= seed = nothing =#
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
    )
    Experiment(
        agent,
        env,
        stop_condition,
        hook,
        "# MAC with CartPole",
    )
end
