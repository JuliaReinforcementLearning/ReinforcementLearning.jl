@testset "PrioritizedDQNLearner" begin
    env = CartPoleEnv(;T=Float32, seed=11)
    ns, na = length(rand(get_observation_space(env))), length(get_action_space(env))
    agent = Agent(
        policy = QBasedPolicy(
            learner = PrioritizedDQNLearner(
                approximator = NeuralNetworkApproximator(
                    model = Chain(
                        Dense(ns, 128, relu;initW=seed_glorot_uniform(seed=17)),
                        Dense(128, 128, relu;initW=seed_glorot_uniform(seed=23)),
                        Dense(128, na;initW=seed_glorot_uniform(seed=39))
                        ) |> gpu,
                    optimizer = ADAM(),
                ),
                target_approximator = NeuralNetworkApproximator(
                    model = Chain(
                        Dense(ns, 128, relu;initW=seed_glorot_uniform(seed=17)),
                        Dense(128, 128, relu;initW=seed_glorot_uniform(seed=23)),
                        Dense(128, na;initW=seed_glorot_uniform(seed=39))
                        ) |> gpu,
                    optimizer = ADAM(),
                ),
                loss_func = huber_loss_unreduced,
                stack_size = nothing,
                batch_size = 32,
                update_horizon=1,
                min_replay_history = 100,
                update_freq = 1,
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

    @info "stats for PrioritizedDQNLearner" avg_reward = mean(hook[1].rewards) avg_fps = 1/mean(hook[2].times)
end

