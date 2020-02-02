@testset "Agent" begin
    action_space = DiscreteSpace(3)
    agent = Agent(;
        policy = RandomPolicy(; action_space = action_space),
        trajectory = VectorialCompactSARTSATrajectory(),
    )

    obs1 = (state = 1,)
    agent(PRE_EPISODE_STAGE, obs1)
    a1 = agent(PRE_ACT_STAGE, obs1)
    @test a1 ∈ action_space

    obs2 = (reward = 1.0, terminal = true, state = 2)
    agent(POST_ACT_STAGE, obs2)
    dummy_action = agent(POST_EPISODE_STAGE, obs2)

    @test length(agent.trajectory) == 1
    @test get_trace(agent.trajectory, :state) == [1]
    @test get_trace(agent.trajectory, :action) == [a1]
    @test get_trace(agent.trajectory, :reward) == [get_reward(obs2)]
    @test get_trace(agent.trajectory, :terminal) == [get_terminal(obs2)]
    @test get_trace(agent.trajectory, :next_state) == [get_state(obs2)]
    @test get_trace(agent.trajectory, :next_action) == [dummy_action]
end
