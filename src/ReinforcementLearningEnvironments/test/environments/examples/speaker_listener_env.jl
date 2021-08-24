@testset "SpeakerListenerEnv" begin
    rng = StableRNG(123)
    env = SpeakerListenerEnv(rng = rng)

    RLBase.test_interfaces!(env)
    RLBase.test_runnable!(env)
end
