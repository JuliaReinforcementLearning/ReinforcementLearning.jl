@testset "agent.jl" begin
    agent = Agent(
        RandomPolicy(),
        Trajectory(
            CircularArraySARTTraces(; capacity = 1_000),
            BatchSampler(1),
            InsertSampleRatioController(),
        ),
    )
    env = RandomWalk1D()
    agent(PreActStage(), env)
    @test state(env) == agent.cache.state
    @test agent(env) in (1,2)
    @test isempty(agent.cache)
    @test length(agent.trajectory.container) == 0 
    agent(PostActStage(), env)
    @test agent.cache.reward == 0. && agent.cache.terminal == false
    agent(PreActStage(), env)
    @test state(env) == agent.cache.state
    @test agent(env) in (1,2)
    @test isempty(agent.cache)
    @test length(agent.trajectory.container) == 1

    #The following tests ensure the args and kwargs are passed to the policy. 
    @test_throws "no method matching (::RandomPolicy" agent(env, 1)
    @test_throws "no method matching (::RandomPolicy" agent(env, fake_kwarg = 1)
end
