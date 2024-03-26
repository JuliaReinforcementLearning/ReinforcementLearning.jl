using Test
using ReinforcementLearningTrajectories
using ReinforcementLearningBase
using DomainSets


@testset "Basic PlayerTuple tests" begin
    nt = PlayerTuple(Player(1) => "test1", Player(2) => "test2")
    @test nt.data == (; Symbol(1) => "test1", Symbol(2) => "test2")
    @test typeof(nt).parameters == typeof(nt.data).parameters
    @test nt[Player(1)] == "test1"
    @test PlayerTuple(Player(i) => i for i in 1:2) == PlayerTuple(Player(1) => 1, Player(2) => 2)

    @test iterate(nt) == ("test1", 2)
    @test iterate(nt, 1) == ("test1", 2)
    collect(iterate(nt))

end

@testset "MultiAgentPolicy" begin
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

    multiagent_policy = MultiAgentPolicy(
        PlayerTuple(
            Player(:Cross) => Agent(RandomPolicy(), trajectory_1),
            Player(:Nought) => Agent(RandomPolicy(), trajectory_2),
        )
    )

    @test multiagent_policy.agents[Player(:Cross)].policy isa RandomPolicy
end

@testset "MultiAgentHook" begin
    env = TicTacToeEnv()
    composed_hook = ComposedHook(
        BatchStepsPerEpisode(10),
        RewardsPerEpisode(),
        StepsPerEpisode(),
        TotalRewardPerEpisode(),
        TimePerStep()
    )

    multiagent_hook = MultiAgentHook(PlayerTuple(Player(:Cross) => composed_hook, Player(:Nought) => EmptyHook()))
    @test multiagent_hook.hooks[Player(:Cross)][3] isa StepsPerEpisode
end

@testset "CurrentPlayerIterator" begin
    env = TicTacToeEnv()
    player_log = []
    i = 0
    for player in RLCore.CurrentPlayerIterator(env)
        i += 1
        push!(player_log, player)
        RLBase.act!(env, 1)
        i == 2 && break
    end
    @test player_log == Player.([:Cross, :Nought])
end

@testset "Basic TicTacToeEnv (Sequential) env checks" begin
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
    hook = StepsPerEpisode()

    @test RLBase.reward(env, Player(:Cross)) == 0
    @test length(RLBase.legal_action_space(env)) == 9
    Base.run(multiagent_policy, env, Sequential(), stop_condition, multiagent_hook)

    @test multiagent_hook.hooks[Player(:Nought)].steps[1] > 0
    @test multiagent_hook.hooks[Player(:Cross)].steps[1] > 0

    @test RLBase.is_terminated(env)
    @test RLEnvs.is_win(env, Player(:Cross)) isa Bool
    @test RLEnvs.is_win(env, Player(:Nought)) isa Bool
    @test RLBase.reward(env, Player(:Cross)) == (RLBase.reward(env, Player(:Nought)) * -1)
    @test RLBase.legal_action_space_mask(env, Player(:Cross)) == falses(9)
    @test RLBase.legal_action_space(env) == []

    @test RLBase.state(env, Observation{BitArray{3}}(), Player(:Cross)) isa BitArray{3}
    @test RLBase.state_space(env, Observation{BitArray{3}}(), Player(:Cross)) isa ArrayProductDomain
    @test RLBase.state_space(env, Observation{String}(), Player(:Cross)) isa DomainSets.FullSpace{String}
    @test RLBase.state(env, Observation{String}(), Player(:Cross)) isa String
    @test RLBase.state(env, Observation{String}()) isa String
end

@testset "next_player!" begin
    env = TicTacToeEnv()
    @test RLBase.next_player!(env) == Player(:Nought)
end

@testset "Basic RockPaperScissors (simultaneous) env checks" begin
    trajectory_1 = Trajectory(
        CircularArraySARTSTraces(; capacity = 1, action = Any => (1,), state = Any => (1,), reward = Any => (2,)),
        BatchSampler(1),
        InsertSampleRatioController(n_inserted = -1),
    )

    trajectory_2 = Trajectory(
        CircularArraySARTSTraces(; capacity = 1, action = Any => (1,), state = Any => (1,), reward = Any => (2,)),
        BatchSampler(1),
        InsertSampleRatioController(n_inserted = -1),
    )

    @test MultiAgentPolicy(PlayerTuple(
        Player(1) => Agent(RandomPolicy(), trajectory_1),
        Player(2) => Agent(RandomPolicy(), trajectory_2),
    )) isa MultiAgentPolicy

    @test MultiAgentPolicy(PlayerTuple(
        Player(1) => Agent(RandomPolicy(), trajectory_1),
        Player(2) => Agent(RandomPolicy(), trajectory_2),
    )) isa MultiAgentPolicy

    multiagent_policy = MultiAgentPolicy(PlayerTuple(
        Player(1) => Agent(RandomPolicy(), trajectory_1),
        Player(2) => Agent(RandomPolicy(), trajectory_2),
    ))

    env = RockPaperScissorsEnv()
    stop_condition = StopIfEnvTerminated()
    composed_hook = ComposedHook(
        BatchStepsPerEpisode(10),
        RewardsPerEpisode(),
        StepsPerEpisode(),
        TotalRewardPerEpisode(),
        TimePerStep()
    )

    multiagent_hook = MultiAgentHook(PlayerTuple(Player(1) => composed_hook, Player(2) => EmptyHook()))

    @test Base.iterate(RLCore.CurrentPlayerIterator(env))[1] == SimultaneousPlayer()
    @test Base.iterate(RLCore.CurrentPlayerIterator(env), env)[1] == SimultaneousPlayer()
    @test Base.iterate(multiagent_policy)[1] isa Agent
    @test Base.iterate(multiagent_policy, 1)[1] isa Agent

    @test Base.getindex(multiagent_policy, Player(1)) isa Agent
    @test Base.getindex(multiagent_hook, Player(1))[3] isa StepsPerEpisode

    @test Base.keys(multiagent_policy) == (Player(1), Player(2))
    @test Base.keys(multiagent_hook) == (Player(1), Player(2))

    @test length(RLBase.legal_action_space(env)) == 9
    Base.run(multiagent_policy, env, stop_condition, multiagent_hook)

    @test multiagent_hook[Player(1)][1].steps[1][1] == 1
    @test -1 <= multiagent_hook[Player(1)][2].rewards[1][1] <= 1
    @test multiagent_hook[Player(1)][3].steps[1] == 1    
    @test -1 <= multiagent_hook[Player(1)][4].rewards[1][1] <= 1
    @test 0 <= multiagent_hook[Player(1)][5].times[1] <= 5

    # Add more hook tests here...

    # TODO: Split up TicTacToeEnv and MultiAgent tests
    @test RLBase.is_terminated(env)
    @test RLBase.legal_action_space(env) == action_space(env)
    @test RLBase.legal_action_space(env, Player(1)) == ()
    @test RLBase.action_space(env, Player(1)) == ('💎', '📃', '✂')
    env = RockPaperScissorsEnv()
    push!(multiagent_policy, PreActStage(), env)
    a = RLBase.plan!(multiagent_policy, env)
    @test [i for i in a][1] ∈ ['💎', '📃', '✂']
    @test [i for i in a][2] ∈ ['💎', '📃', '✂']
    @test RLBase.act!(env, a)
end

@testset "Sequential Environments correctly ended by termination signal" begin
    #rng = StableRNGs.StableRNG(123)
    e = TicTacToeEnv();
    m = MultiAgentPolicy(PlayerTuple(player => RandomPolicy() for player in players(e)))
    hooks = MultiAgentHook(PlayerTuple(p => EmptyHook() for p ∈ players(e)))

    let err = nothing
        try
            x = run(m, e, StopAfterNEpisodes(10), hooks)
        catch err
        end
        @test !(err isa Exception)
    end
end
