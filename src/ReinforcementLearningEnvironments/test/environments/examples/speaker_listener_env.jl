@testset "SpeakerListenerEnv" begin
    env = SpeakerListenerEnv()

    RLBase.test_interfaces!(env)
    RLBase.test_runnable!(env)
end
