@testset "TinayHanabiEnv" begin

    env = TinyHanabiEnv()

    RLBase.test_interfaces!(env)
    RLBase.test_runnable!(env)

end
