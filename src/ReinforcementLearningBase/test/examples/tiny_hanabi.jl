@testset "TinayHanabiEnv" begin

    env = TinyHanabiEnv()

    RLBase.test_interfaces!(env)

end
