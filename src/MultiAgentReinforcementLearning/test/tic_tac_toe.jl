@testset "Tic tac toe" begin
    e = TicTacToeEnv()
    m = MultiAgentManager(Dict(player =>
        Agent(
            RandomPolicy(),
            Trajectory(
                CircularArraySARTTraces(; capacity = 1_000),
                DummySampler(),
        ),
    ) for player in players(e)), current_player(e))
    hook = RewardsPerEpisode()
    out = run(RandomPolicy(), e, StopAfterEpisode(50), hook)
end
