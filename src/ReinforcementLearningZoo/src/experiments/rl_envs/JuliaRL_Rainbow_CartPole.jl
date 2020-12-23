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
    rng = StableRNG(seed)

    env = CartPoleEnv(; T = Float32, rng = rng)
    ns, na = length(state(env)), length(action_space(env))

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
                loss_func = (ŷ, y) -> logitcrossentropy(ŷ, y; agg = identity),
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
        trajectory = CircularArrayPSARTTrajectory(
            capacity = 1000,
            state = Vector{Float32} => (ns,),
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
    )

    description = """
    This experiment uses the `RainbowLearner` method with three dense layers to approximate the distributed Q value.
    The testing environment is CartPoleEnv.
    """

    Experiment(agent, env, stop_condition, hook, description)
end
