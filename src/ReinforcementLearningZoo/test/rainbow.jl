@testset "RainbowLearner" begin
    env = CartPoleEnv(;T=Float32, seed=11)
    ns, na = length(rand(get_observation_space(env))), length(get_action_space(env))
    n_atoms = 51
    agent = Agent(
        policy = QBasedPolicy(
            learner = RainbowLearner(
                approximator = NeuralNetworkApproximator(
                    model = Chain(
                        Dense(ns, 128, relu;initW=seed_glorot_uniform(seed=17)),
                        Dense(128, 128, relu;initW=seed_glorot_uniform(seed=23)),
                        Dense(128, na*n_atoms;initW=seed_glorot_uniform(seed=39))
                        ) |> cpu,
                    optimizer = ADAM(0.0005),
                ),
                target_approximator = NeuralNetworkApproximator(
                    model = Chain(
                        Dense(ns, 128, relu;initW=seed_glorot_uniform(seed=17)),
                        Dense(128, 128, relu;initW=seed_glorot_uniform(seed=23)),
                        Dense(128, na*n_atoms;initW=seed_glorot_uniform(seed=39))
                        ) |> cpu,
                    optimizer = ADAM(0.0005),
                ),
                n_actions = na,
                n_atoms = n_atoms,
                Vₘₐₓ=200.0f0,
                Vₘᵢₙ=0.0f0,
                update_freq = 1,
                γ = 0.99f0,
                update_horizon=1,
                batch_size = 32,
                stack_size = nothing,
                min_replay_history = 100,
                loss_func = logitcrossentropy_unreduced,
                target_update_freq=100,
                seed = 22
            ),
            explorer = EpsilonGreedyExplorer(kind=:exp, ϵ_stable = 0.01, decay_steps = 500, seed=33),
        ),
        trajectory = CircularCompactPSARTSATrajectory(
            capacity = 1000,
            state_type = Float32,
            state_size = (ns,),
        )
    )
    hook = ComposedHook(TotalRewardPerEpisode(), TimePerStep())
    run(agent, env, StopAfterStep(10000), hook)

    @info "stats for RainbowLearner" avg_reward = mean(hook[1].rewards) avg_fps = 1/mean(hook[2].times)
end
