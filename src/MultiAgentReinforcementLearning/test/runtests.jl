using Test
using ReinforcementLearningEnvironments
using ReinforcementLearningTrajectories
using ReinforcementLearningCore
using ReinforcementLearningBase
using MultiAgentReinforcementLearning
TicTacToeEnv([1 0 1; 0 1 1; 1 1 1;;; 0 0 0; 1 0 0; 0 0 0;;; 0 1 0; 0 0 0; 0 0 0], :Nought)

@testset "MultiAgentPolicy" begin
    trajectory_1 = Trajectory(
        CircularArraySARTTraces(; capacity = 1),
        BatchSampler(1),
        InsertSampleRatioController(n_inserted = -1),
    )

    trajectory_2 = Trajectory(
        CircularArraySARTTraces(; capacity = 1),
        BatchSampler(1),
        InsertSampleRatioController(n_inserted = -1),
    )

    multiagent_policy = MultiAgentPolicy((;
        :Cross => Agent(RandomPolicy(), trajectory_1),
        :Nought => Agent(RandomPolicy(), trajectory_2),
    ))

    @test multiagent_policy.agents[:Cross].policy isa RandomPolicy
end

@testset "MultiAgentHook" begin
    multiagent_hook = MultiAgentHook((; :Cross => StepsPerEpisode(), :Nought => StepsPerEpisode()))
    @test multiagent_hook.hooks[:Cross] isa StepsPerEpisode
end

@testset "CurrentPlayerIterator" begin
    env = TicTacToeEnv()
    player_log = []
    i = 0
    for player in CurrentPlayerIterator(env)
        i += 1
        push!(player_log, player)
        env(1)
        i == 2 && break
    end
    @test player_log == [:Cross, :Nought]
end


# import MultiAgentReinforcementLearning: MultiAgentHook
@testset "Basic TicTacToeEnv (Sequential) env checks" begin
    trajectory_1 = Trajectory(
        CircularArraySARTTraces(; capacity = 1),
        BatchSampler(1),
        InsertSampleRatioController(n_inserted = -1),
    )

    trajectory_2 = Trajectory(
        CircularArraySARTTraces(; capacity = 1),
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
    hook = StepsPerEpisode()

    @test length(RLBase.legal_action_space(env)) == 9
    run(multiagent_policy, env, stop_condition, multiagent_hook)
    # TODO: Split up TicTacToeEnv and MultiAgent tests
    @test RLBase.is_terminated(env)
    @test RLEnvs.is_win(env, :Cross) != RLEnvs.is_win(env, :Nought)
    @test RLBase.legal_action_space(env) == []
end


@testset "Basic RockPaperScissors (simultaneous) env checks" begin
    trajectory_1 = Trajectory(
        CircularArraySARTTraces(; capacity = 1, action = Any => (2,)),
        BatchSampler(1),
        InsertSampleRatioController(n_inserted = -1),
    )

    trajectory_2 = Trajectory(
        CircularArraySARTTraces(; capacity = 1, action = Any => (2,)),
        BatchSampler(1),
        InsertSampleRatioController(n_inserted = -1),
    )

    multiagent_policy = MultiAgentPolicy((;
        Symbol(1) => Agent(RandomPolicy(), trajectory_1),
        Symbol(2) => Agent(RandomPolicy(), trajectory_2),
    ))

    env = RockPaperScissorsEnv()
    stop_condition = StopWhenDone()
    multiagent_hook = MultiAgentHook((; Symbol(1) => StepsPerEpisode(), Symbol(2) => StepsPerEpisode()))

    @test Base.iterate(env)[1] == SimultaneousPlayer()
    @test Base.iterate(env, env)[1] == SimultaneousPlayer()
    @test Base.iterate(multiagent_policy)[1] isa Agent
    @test Base.iterate(multiagent_policy, 1)[1] isa Agent
    
    @test Base.getindex(multiagent_policy, Symbol(1)) isa Agent
    @test Base.getindex(multiagent_hook, Symbol(1)) isa StepsPerEpisode

    @test Base.keys(multiagent_policy) == (Symbol(1), Symbol(2))
    @test Base.keys(multiagent_hook) == (Symbol(1), Symbol(2))

    @test length(RLBase.legal_action_space(env)) == 9
    run(multiagent_policy, env, stop_condition, multiagent_hook)
    # TODO: Split up TicTacToeEnv and MultiAgent tests
    @test RLBase.is_terminated(env)
    @test RLBase.legal_action_space(env) == ()

    env = RockPaperScissorsEnv()
    (multiagent_policy)(PreActStage(), env)
    # multiagent_policy(env)
    @test [multiagent_policy(env)...] == [('📃', '✂'), ('💎', '✂')]
end



Base.keys(p::MultiAgentPolicy) = keys(p.agents)
Base.keys(p::MultiAgentHook) = keys(p.hooks)
