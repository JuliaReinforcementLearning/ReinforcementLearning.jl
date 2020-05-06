@testset "base" begin
    env = LotteryEnv(; seed = 222)
    action_space = get_action_space(env)
    policy = RandomPolicy(env; seed = 123)
    reset!(env)
    run(policy, env)
    @test get_terminal(observe(env))
end
