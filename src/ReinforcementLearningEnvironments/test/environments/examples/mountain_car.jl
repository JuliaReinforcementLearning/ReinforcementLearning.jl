@testset "mountain_car" begin

    env = MountainCarEnv(; rng = StableRNG(123))
    RLBase.test_interfaces!(env)
    RLBase.test_runnable!(env)

    env = ContinuousMountainCarEnv(; rng = StableRNG(123))
    RLBase.test_interfaces!(env)
    RLBase.test_runnable!(env)

end
