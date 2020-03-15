@testset "A2CGAE" begin
N_ENV = 16
UPDATE_FREQ = 10
env = MultiThreadEnv([CartPoleEnv(;T=Float32, seed=i) for i in 1:N_ENV])
ns, na = length(rand(get_observation_space(env[1]))), length(get_action_space(env[1]))
reset!(env, is_force=true)
agent = Agent(
    policy = QBasedPolicy(
        learner=A2CGAELearner(
            approximator=NeuralNetworkApproximator(
                model=ActorCritic(
                    actor=Chain(
                        Dense(ns, 256, relu;initW = seed_glorot_uniform(seed = 17)),
                        Dense(256, na;initW = seed_glorot_uniform(seed = 23)),
                        softmax
                    ),
                    critic=Chain(
                        Dense(ns, 256, relu;initW = seed_glorot_uniform(seed = 29)),
                        Dense(256, 1;initW = seed_glorot_uniform(seed = 29)),
                    )
                ) |> gpu,
                optimizer=ADAM(1e-3),
                kind=HYBRID_APPROXIMATOR
            ),
            γ=0.99f0,
            λ=0.97f0,
            actor_loss_weight = 1.0f0,
            critic_loss_weight = 0.5f0,
            entropy_loss_weight = 0.001f0
        ),
        explorer = BatchExplorer((WeightedExplorer(;is_normalized=true, seed=s) for s in 10:9+N_ENV)...)
    ),
    trajectory = CircularCompactSARTSATrajectory(
        ;capacity=UPDATE_FREQ,
        state_type=Float32,
        state_size=(ns, N_ENV),
        action_type=Int,
        action_size=(N_ENV,),
        reward_type=Float32,
        reward_size=(N_ENV,),
        terminal_type=Bool,
        terminal_size=(N_ENV,)
    )
)
hook = TotalBatchRewardPerEpisode(N_ENV)
run(agent, env, StopAfterStep(100000), hook)
@info "stats for A2CGAELearner" avg_reward = mean(Iterators.flatten(hook.rewards))
end
