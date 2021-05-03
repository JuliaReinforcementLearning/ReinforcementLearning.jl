@testset "TicTacToeEnv" begin

    env = TicTacToeEnv()

    RLBase.test_interfaces!(env)
    RLBase.test_runnable!(env)

    @test length(state_space(env, Observation{Int}())) == 5478

end
