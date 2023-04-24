@testset "TicTacToeEnv" begin
    using DomainSets
    using ReinforcementLearningEnvironments, ReinforcementLearningBase
    env = TicTacToeEnv()

    RLBase.test_interfaces!(env)
    RLBase.test_runnable!(env)

    @test length(state_space(env, Observation{Int}())) == 5478

    @test RLBase.state(env, Observation{BitArray{3}}(), Symbol(1)) == env.board
    @test RLBase.state_space(env, Observation{BitArray{3}}(), Symbol(1)) isa ArrayProductDomain
    @test RLBase.state_space(env, Observation{String}(), Symbol(1)) isa DomainSets.FullSpace{String}
    @test RLBase.state(env, Observation{String}(), Symbol(1)) isa String
end

