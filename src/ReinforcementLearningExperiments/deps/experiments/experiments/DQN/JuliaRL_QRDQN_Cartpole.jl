function Experiment(
    ::Val{:JuliaRL},
    ::Val{:QRDQN},
    ::Val{:CartPole},
    ::Nothing;
    save_dir=nothing,
    seed=123,
)

    N = 10

    rng = StableRNG(seed)
    env = CartPoleEnv(; T=Float32, rng=rng)
    ns, na = length(state(env)), length(action_space(env))

    init = glorot_uniform(rng)

    agent = Agent(
        policy=QBasedPolicy(
            learner=QRDQNLearner(
                approximator=NeuralNetworkApproximator(
                    model=Chain(
                        Dense(ns, 128, relu; init = init),
                        Dense(128, 128, relu; init = init),
                        Dense(128, N * na; init = init),
                    ) |> cpu,
                    optimizer=ADAM(),
                ),
                target_approximator=NeuralNetworkApproximator(
                    model=Chain(
                        Dense(ns, 128, relu; init = init),
                        Dense(128, 128, relu; init = init),
                        Dense(128, N * na; init = init),
                    ) |> cpu,
                ),
                stack_size=nothing,
                batch_size=32,
                update_horizon=1,
                min_replay_history=100,
                update_freq=1,
                target_update_freq=100,
                n_quantile=N,
                rng=rng,
            ),
            explorer=EpsilonGreedyExplorer(
                kind=:exp,
                ϵ_stable=0.01,
                decay_steps=500,
                rng=rng,
            ),
        ),
        trajectory=CircularArraySARTTrajectory(
            capacity=1000,
            state=Vector{Float32} => (ns,),
        ),
    )

    stop_condition = StopAfterStep(10_000)

    hook = ComposedHook(
        TotalRewardPerEpisode(),
        TimePerStep(),
    )

    description = """
    This experiment uses the `QRDQNLearner` method with three dense layers to approximate the quantile values.
    The testing environment is CartPoleEnv.
    """

    Experiment(agent, env, stop_condition, hook, description)
end
