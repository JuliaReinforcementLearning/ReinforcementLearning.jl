@testset "cartpole_env" begin

    env = CartPoleEnv(; rng = MersenneTwister(123))
    RLBase.test_interfaces!(env)
    RLBase.test_runnable!(env)

    env = CartPoleEnv(;T=Float32, rng = MersenneTwister(123), thetathreshold = 90.0)
    RLBase.test_interfaces!(env)
    RLBase.test_runnable!(env)

end
