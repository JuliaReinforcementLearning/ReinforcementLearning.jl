using ReinforcementLearningBase, ReinforcementLearningEnvironments
using ReinforcementLearningCore: SRT, update!
using ReinforcementLearningCore

@testset "agent.jl" begin
    @testset "Agent Cache struct" begin
        srt = SRT{Int64, Float64, Bool}()
        update!(srt, 2)
        @test srt.state == 2
        update!(srt, 1.0, true)
        @test srt.reward == 1.0
        @test srt.terminal == true
    end

    @testset "Trajectory SART struct compatibility" begin
        srt_1 = SRT()
        srt_2 = SRT{Any, Nothing, Nothing}()
        srt_2.state = 1
        srt_3 = SRT{Any, Any, Bool}()
        srt_3.state = 1
        srt_3.reward = 1.0
        srt_3.terminal = true
        trajectory = Trajectory(
            CircularArraySARTTraces(; capacity = 1_000, reward=Float64=>()),
            DummySampler(),
        )
        
        @test_throws ArgumentError push!(trajectory, srt_1)
        push!(trajectory, srt_2, 1)
        @test length(trajectory.container) == 0
        push!(trajectory, srt_3, 2)
        @test length(trajectory.container) == 1
        @test trajectory.container[:action] == [1]
        push!(trajectory, srt_3, 3)
        @test trajectory.container[:action] == [1, 2]
        @test trajectory.container[:state] == [1, 1]
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
                @test agent.cache.state != nothing
                @test agent.cache.reward == nothing
                @test agent.cache.terminal == nothing
                @test state(env) == agent.cache.state
                @test agent(env) in (1,2)
                @test length(agent.trajectory.container) == 0 
                agent(PostActStage(), env)
                @test agent.cache.reward == 0. && agent.cache.terminal == false
                agent(PreActStage(), env)
                @test state(env) == agent.cache.state
                @test agent(env) in (1,2)
                @test length(agent.trajectory.container) == 1

                #The following tests ensure the args and kwargs are passed to the policy. 
                @test_throws "no method matching (::RandomPolicy" agent(env, 1)
                @test_throws "no method matching (::RandomPolicy" agent(env, fake_kwarg = 1)
            end

            @testset "Test update! method" begin
                env = RandomWalk1D()
                agent = agent_list[i]
                agent(PostActStage(), env)
                update!(agent, 7)
                @test agent.cache.state == 7
                RLBase.reset!(agent.cache)
                @test agent.cache.state == nothing
            end 
        end
    end
end


