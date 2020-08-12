@testset "simple workflow" begin
    env = CartPoleEnv{Float32}() |> StateOverriddenEnv(deepcopy)
    agent = Agent(;
        policy = RandomPolicy(env),
        trajectory = CircularCompactSARTSATrajectory(;capacity=10_000, state_type = Float32, state_size=(4,)),
    )
    N_EPISODE = 10000
    hook = TotalRewardPerEpisode()
    run(agent, env, StopAfterEpisode(N_EPISODE), hook)

    @test isapprox(sum(hook.rewards) / N_EPISODE, 21; atol = 2)
end
