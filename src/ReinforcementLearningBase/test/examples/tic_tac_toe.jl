@testset "TicTacToeEnv" begin

    env = TicTacToeEnv()

    RLBase.test_interfaces(env)

    @test length(state_space(env, Observation{Int}())) == 5478

end
