@testset "simple workflow" begin
    env = StateTransformedEnv(CartPoleEnv{Float32}(); state_mapping = deepcopy)
    policy = RandomPolicy(action_space(env))
    N_EPISODE = 10_000
    hook = TotalRewardPerEpisode()
    run(policy, env, StopAfterEpisode(N_EPISODE), hook)

    @test isapprox(sum(hook[]) / N_EPISODE, 21; atol = 2)
end

@testset "multi agent" begin
    # https://github.com/JuliaReinforcementLearning/ReinforcementLearning.jl/issues/393
    rps = RockPaperScissorsEnv() |> SequentialEnv
    ma_policy =
        MultiAgentManager((NamedPolicy(p => RandomPolicy()) for p in players(rps))...)

    run(ma_policy, rps, StopAfterEpisode(10))
end
