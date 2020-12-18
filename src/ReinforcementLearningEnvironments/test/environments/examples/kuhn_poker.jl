@testset "KuhnPokerEnv" begin

    env = KuhnPokerEnv()
    RLBase.test_interfaces!(env)
    RLBase.test_runnable!(env)

end
