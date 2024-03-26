@testset "TicTacToeEnv" begin
    using DomainSets
    using ReinforcementLearningEnvironments, ReinforcementLearningBase, ReinforcementLearningCore

    trajectory_1 = Trajectory(
        CircularArraySARTSTraces(; capacity = 1),
        BatchSampler(1),
        InsertSampleRatioController(n_inserted = -1),
    )

    trajectory_2 = Trajectory(
        CircularArraySARTSTraces(; capacity = 1),
        BatchSampler(1),
        InsertSampleRatioController(n_inserted = -1),
    )

    multiagent_policy = MultiAgentPolicy(PlayerTuple(
        Player(:Cross) => Agent(RandomPolicy(), trajectory_1),
        Player(:Nought) => Agent(RandomPolicy(), trajectory_2),
    ))

    multiagent_hook = MultiAgentHook(PlayerTuple(Player(:Cross) => StepsPerEpisode(), Player(:Nought) => StepsPerEpisode()))

    env = TicTacToeEnv()
    stop_condition = StopIfEnvTerminated()

    RLBase.test_interfaces!(env)
    RLBase.test_runnable!(env)

    @test length(state_space(env, Observation{Int}())) == 5478

    @test RLBase.state(env, Observation{BitArray{3}}(), Player(:Cross)) == env.board
    @test RLBase.state_space(env, Observation{BitArray{3}}(), Player(:Cross)) isa ArrayProductDomain
    @test RLBase.state_space(env, Observation{String}(), Player(:Cross)) isa DomainSets.FullSpace{String}
    @test RLBase.state(env, Observation{String}(), Player(:Cross)) isa String
    @test RLBase.state(env, Observation{String}()) isa String
    Base.run(multiagent_policy, env, stop_condition, multiagent_hook)
    @test RLBase.legal_action_space_mask(env, Player(:Cross)) == falses(9)
end
