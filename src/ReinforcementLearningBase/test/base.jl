@testset "base" begin
    env = LotteryEnv()
    Random.seed!(env, 123)
    policy = RandomPolicy(env)
    Random.seed!(policy, 111)
    reset!(env)
    run(policy, env)
    @test get_terminal(env)
end
