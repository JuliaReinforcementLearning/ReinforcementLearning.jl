@testset "simple workflow" begin
    env = CartPoleEnv{Float32}() |> StateOverriddenEnv(deepcopy)
    agent = Agent(;
        policy = RandomPolicy(env),
        trajectory = VectorialCompactSARTSATrajectory(; state_type = Vector{Float32}),
    )
    N_EPISODE = 10000
    hook = TotalRewardPerEpisode()
    run(agent, env, StopAfterEpisode(N_EPISODE), hook)

    @test isapprox(sum(hook.rewards) / N_EPISODE, 21; atol = 2)
end
