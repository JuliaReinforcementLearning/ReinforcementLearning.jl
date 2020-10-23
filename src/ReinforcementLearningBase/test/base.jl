@testset "base" begin
    env = LotteryEnv()
    Random.seed!(env, 123)
    policy = RandomPolicy(env)
    Random.seed!(policy, 111)
    reset!(env)
    run(policy, env)
    @test get_terminal(env)

    discrete_env = ActionTransformedEnv(
        a -> get_actions(env)[a];  # action index to action
        mapping=x -> Dict(x => i for (i, a) in enumerate(get_actions(env)))[x], # arbitrary vector to DiscreteSpace
    )(
        env
    )
    policy = RandomPolicy(discrete_env)
    reset!(discrete_env)
    run(policy, discrete_env)
    @test get_terminal(discrete_env)
end
