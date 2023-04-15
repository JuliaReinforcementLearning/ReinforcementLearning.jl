using Test
using ReinforcementLearningEnvironments
using ReinforcementLearningTrajectories
using ReinforcementLearningCore
using ReinforcementLearningBase
using MultiAgentReinforcementLearning

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

multiagent_policy = MultiAgentPolicy((Cross = Agent(RandomPolicy(), trajectory_1), Nought = Agent(RandomPolicy(), trajectory_2)))

env = TicTacToeEnv()
stop_condition = StopWhenDone()
hook = StepsPerEpisode()
run(multiagent_policy, env, stop_condition, hook)
RLBase.is_terminated(env)

@testset "Basic TicTacToeEnv env checks" begin
    # TODO: Split up TicTacToeEnv and MultiAgent tests
    @test RLEnvs.is_win(env, :Cross) != RLEnvs.is_win(env, :Nought)
end
