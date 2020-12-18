@testset "acrobot_env" begin

    env = AcrobotEnv(; rng = MersenneTwister(123))
    RLBase.test_interfaces!(env)
    RLBase.test_runnable!(env)

end
