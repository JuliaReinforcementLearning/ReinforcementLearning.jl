@testset "TicTacToeEnv" begin
    using DomainSets
    using ReinforcementLearningEnvironments, ReinforcementLearningBase, ReinforcementLearningCore

    trajectory_1 = Trajectory(
        CircularArraySARSTTraces(; capacity = 1),
        BatchSampler(1),
        InsertSampleRatioController(n_inserted = -1),
    )

    trajectory_2 = Trajectory(
        CircularArraySARSTTraces(; capacity = 1),
        BatchSampler(1),
        InsertSampleRatioController(n_inserted = -1),
    )

    multiagent_policy = MultiAgentPolicy((;
        :Cross => Agent(RandomPolicy(), trajectory_1),
        :Nought => Agent(RandomPolicy(), trajectory_2),
    ))

    multiagent_hook = MultiAgentHook((; :Cross => StepsPerEpisode(), :Nought => StepsPerEpisode()))

    env = TicTacToeEnv()
    stop_condition = StopWhenDone()

    RLBase.test_interfaces!(env)
    RLBase.test_runnable!(env)

    @test length(state_space(env, Observation{Int}())) == 5478

    @test RLBase.state(env, Observation{BitArray{3}}(), :Cross) == env.board
    @test RLBase.state_space(env, Observation{BitArray{3}}(), :Cross) isa ArrayProductDomain
    @test RLBase.state_space(env, Observation{String}(), :Cross) isa DomainSets.FullSpace{String}
    @test RLBase.state(env, Observation{String}(), :Cross) isa String
    @test RLBase.state(env, Observation{String}()) isa String
    Base.run(multiagent_policy, env, stop_condition, multiagent_hook)
    @test RLBase.legal_action_space_mask(env, :Cross) == falses(9)
end
