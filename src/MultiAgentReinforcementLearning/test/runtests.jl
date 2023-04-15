using Test
using ReinforcementLearningEnvironments
using ReinforcementLearningTrajectories
using ReinforcementLearningCore

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

env = TicTacToeEnv1()
stop_condition = StopWhenDone()
hook = StepsPerEpisode()
run(multiagent_policy, env, stop_condition, hook)
is_terminated(env)


# TODO: Split up TicTacToeEnv and MultiAgent tests
is_win(env, :Cross)

is_win(env, :Nought)
