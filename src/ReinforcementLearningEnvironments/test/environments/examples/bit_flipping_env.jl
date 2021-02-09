@testset "bit_flipping_env" begin
    rng = StableRNG(123)
    env = BitFlippingEnv(; N = 7, T = 2, rng = rng)
    test_state = state(env, GoalState())
    env(1)
    @test is_terminated(env) == false
    env(1)
    @test is_terminated(env) == true
    RLBase.test_interfaces!(env)
    RLBase.test_runnable!(env)

end
