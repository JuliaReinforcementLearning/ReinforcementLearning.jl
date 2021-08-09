@testset "" begin
    rng = StableRNG(123)
    env = GraphShortestPathEnv(rng)

    RLBase.test_interfaces!(env)
    RLBase.test_runnable!(env)

end

