@testset "TicTacToeEnv" begin
    using DomainSets
    using ReinforcementLearningEnvironments, ReinforcementLearningBase, ReinforcementLearningCore

    env = TicTacToeEnv()
    stop_condition = StopWhenDone()
    multiagent_hook = MultiAgentHook((; Symbol(1) => StepsPerEpisode(), Symbol(2) => StepsPerEpisode()))
    multiagent_policy = MultiAgentPolicy((;
        Symbol(1) => Agent(RandomPolicy(), trajectory_1),
        Symbol(2) => Agent(RandomPolicy(), trajectory_2),
    ))

    RLBase.test_interfaces!(env)
    RLBase.test_runnable!(env)

    @test length(state_space(env, Observation{Int}())) == 5478

    @test RLBase.state(env, Observation{BitArray{3}}(), Symbol(1)) == env.board
    @test RLBase.state_space(env, Observation{BitArray{3}}(), Symbol(1)) isa ArrayProductDomain
    @test RLBase.state_space(env, Observation{String}(), Symbol(1)) isa DomainSets.FullSpace{String}
    @test RLBase.state(env, Observation{String}(), Symbol(1)) isa String
    @test RLBase.state(env, Observation{String}()) isa String
    Base.run(multiagent_policy, env, stop_condition, multiagent_hook)
    @test RLBase.legal_action_space_mask(env, Symbol(1)) == falses(9)
end


