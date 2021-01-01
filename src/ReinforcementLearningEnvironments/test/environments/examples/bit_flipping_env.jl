@testset "bit_flipping_env" begin
    rng = StableRNG(123)
    env = BitFlippingEnv(; N = 7, rng = rng)
    test_state = state(env, GoalState())
    RLBase.test_interfaces!(env)
    RLBase.test_runnable!(env)

end
