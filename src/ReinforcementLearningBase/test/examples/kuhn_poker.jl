@testset "KuhnPokerEnv" begin

    env = KuhnPokerEnv()

RLBase.test_interfaces!(env)

end
