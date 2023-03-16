using ReinforcementLearningBase, ReinforcementLearningEnvironments
using ReinforcementLearningCore: AgentCache

using JET, ReinforcementLearningCore, Test
@testset "agent.jl" begin
    @testset "Agent Cache struct" begin
        @test typeof(AgentCache()) == AgentCache{Any, Any, Any}
        env = RandomWalk1D()
        policy = RandomPolicy()
        cache_1 = AgentCache(policy, env)
        @test typeof(cache_1) == AgentCache{Int64, Int64, Float64}
        @test ismissing(cache_1.action)
        @test ismissing(cache_1.state)
        @test ismissing(cache_1.reward)
        @test ismissing(cache_1.terminal)

        cache_1.action = 1
        cache_1.state = 10
        cache_1.reward = 10
        cache_1.terminal = true
        @test cache_1.action == 1
        @test cache_1.state == 10
        @test cache_1.reward == 10.0
        @test cache_1.terminal == true

        RLCore.reset!(cache_1)
        @test ismissing(cache_1.action)
        @test ismissing(cache_1.state)
        @test ismissing(cache_1.reward)
        @test ismissing(cache_1.terminal)


        cache_1.action = 1
        @test (action = 1,) == RLCore.struct_to_namedtuple(cache_1)
        RLCore.reset!(cache_1)
        @test (;) == RLCore.struct_to_namedtuple(cache_1)
    end

    a_1 = Agent(
        RandomPolicy(),
        Trajectory(
            CircularArraySARTTraces(; capacity = 1_000),
            DummySampler(),
        ),
    )
    a_2 = Agent(
        RandomPolicy(),
        Trajectory(
            CircularArraySARTTraces(; capacity = 1_000),
            BatchSampler(1),
            InsertSampleRatioController(),
        ),
    )
    a_3 = Agent(
        RandomPolicy(),
        Trajectory(
            CircularArraySARTTraces(; capacity = 1_000),
            DummySampler(),
        ),
        env=RandomWalk1D()
    )

    agent_list = (a_1, a_2, a_3)
    for i in 1:length(agent_list)
        @testset "Test Agent $i" begin
            env = RandomWalk1D()
            agent = agent_list[i]
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
    end
end
