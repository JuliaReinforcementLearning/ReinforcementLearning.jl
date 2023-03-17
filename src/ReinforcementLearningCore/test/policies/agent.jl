using ReinforcementLearningBase, ReinforcementLearningEnvironments
using ReinforcementLearningCore: SART, sart_to_tuple
using ReinforcementLearningCore

@testset "agent.jl" begin
    @testset "Agent Cache struct" begin
        @test typeof(SART()) == SART{Any, Any, Any}
        env = RandomWalk1D()
        policy = RandomPolicy()
        cache_1 = SART(policy, env)
        @test typeof(cache_1) == SART{Int64, Int64, Float64}
        cache_1.state = 10
        cache_1.action = 1
        cache_1.reward = 10
        cache_1.terminal = true
        @test cache_1.action == 1
        @test cache_1.state == 10
        @test cache_1.reward == 10.0
        @test cache_1.terminal == true

        RLCore.reset!(cache_1)
        @test ismissing(cache_1.state)
        @test ismissing(cache_1.action)
        @test ismissing(cache_1.reward)
        @test ismissing(cache_1.terminal)

        env.pos = 2
        RLCore.update_state!(cache_1, env)
        @test cache_1.state == 2

        env.pos = 1
        RLCore.update_reward!(cache_1, env)
        @test cache_1.reward == -1

        cache_1.state = 1
        cache_1.action = 1
        cache_1.terminal = true
        @test RLCore.sart_to_tuple(cache_1) == (state = 1, action = 1, reward = -1.0, terminal = true)
    end

    @testset "Trajectory SART struct compatibility" begin
        trajectory = Trajectory(
            CircularArraySARTTraces(; capacity = 1_000, reward=Float64=>()),
            DummySampler(),
        )

        sart = SART()
        sart.state = 1
        sart.action = 1
        push!(trajectory, sart_to_tuple(sart))
        sart.reward = 1.0
        sart.terminal = true
        push!(trajectory, sart_to_tuple(sart))
        @test length(trajectory.container) == 1
    end

    @testset "Agent Tests" begin
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

        agent_list = (a_1, a_2)
        for i in 1:length(agent_list)
            @testset "Test Agent $i" begin
                agent = agent_list[i]
                env = RandomWalk1D()
                agent(PreActStage(), env)
                @test ismissing(agent.cache.action)
                @test state(env) == agent.cache.state
                @test agent(env) in (1,2)
                @test ismissing(agent.cache.action)
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
end
