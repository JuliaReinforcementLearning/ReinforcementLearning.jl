@testset "pendulum" begin

    env = PendulumEnv(; rng = StableRNG(123))
    RLBase.test_interfaces!(env)
    RLBase.test_runnable!(env)

    env = PendulumNonInteractiveEnv(; rng = StableRNG(123))
    RLBase.test_interfaces!(env)
    RLBase.test_runnable!(env)

end
