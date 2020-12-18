@testset "cartpole_env" begin

env = CartPoleEnv(;rng=MersenneTwister(123))
RLBase.test_interfaces!(env)
RLBase.test_runnable!(env)

end
