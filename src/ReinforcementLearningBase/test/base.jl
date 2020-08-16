@testset "base" begin
    env = LotteryEnv()
    Random.seed!(env, 123)
    policy = RandomPolicy(env)
    Random.seed!(policy, 111)
    reset!(env)
    run(policy, env)
    @test get_terminal(env)

    policy = TabularRandomPolicy()
    Random.seed!(policy, 111)
    inner_env = LotteryEnv()
    env = inner_env |> ActionTransformedEnv(a -> get_actions(inner_env)[a])
    reset!(env)
    run(policy, env)
    @test get_terminal(env)
end
