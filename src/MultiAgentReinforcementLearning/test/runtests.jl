using Test
using ReinforcementLearningEnvironments
using ReinforcementLearningTrajectories
using ReinforcementLearningCore
using ReinforcementLearningBase
using MultiAgentReinforcementLearning

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

    multiagent_policy = MultiAgentPolicy((
        Cross = Agent(RandomPolicy(), trajectory_1),
        Nought = Agent(RandomPolicy(), trajectory_2),
    ))

    env = TicTacToeEnv()
    stop_condition = StopWhenDone()
    hook = StepsPerEpisode()

    @test length(RLBase.legal_action_space(env)) == 9
    run(multiagent_policy, env, stop_condition, hook)
    # TODO: Split up TicTacToeEnv and MultiAgent tests
    @test RLBase.is_terminated(env)
    @test RLEnvs.is_win(env, :Cross) != RLEnvs.is_win(env, :Nought)
    @test RLBase.legal_action_space(env) == []
end


@testset "Basic RockPaperScissors (simultaneous) env checks" begin
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
        Symbol(1) => Agent(RandomPolicy(), trajectory_1),
        Symbol(2) => Agent(RandomPolicy(), trajectory_2),
    ))

    env = RockPaperScissorsEnv()
    stop_condition = StopWhenDone()
    hook = StepsPerEpisode()

    @test length(RLBase.legal_action_space(env)) == 9
    run(multiagent_policy, env, stop_condition, hook)
    # TODO: Split up TicTacToeEnv and MultiAgent tests
    @test RLBase.is_terminated(env)
    @test RLEnvs.is_win(env, :Cross) != RLEnvs.is_win(env, :Nought)
    @test RLBase.legal_action_space(env) == []
end
