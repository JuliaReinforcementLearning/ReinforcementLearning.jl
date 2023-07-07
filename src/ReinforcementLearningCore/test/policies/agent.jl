using ReinforcementLearningBase, ReinforcementLearningEnvironments
using ReinforcementLearningCore: SRT
using ReinforcementLearningCore

@testset "agent.jl" begin
    @testset "Agent Tests" begin
        a_1 = Agent(
            RandomPolicy(),
            Trajectory(
                CircularArraySARTSTraces(; capacity = 1_000),
                DummySampler(),
            ),
        )
        a_2 = Agent(
            RandomPolicy(),
            Trajectory(
                CircularArraySARTSTraces(; capacity = 1_000),
                BatchSampler(1),
                InsertSampleRatioController(),
            ),
        )

        agent_list = (a_1, a_2)
        for i in 1:length(agent_list)
            @testset "Test Agent $i" begin
                agent = agent_list[i]
                env = RandomWalk1D()
                push!(agent, PreEpisodeStage(), env)
                action = RLBase.plan!(agent, env)
                @test action in (1,2)
                @test length(agent.trajectory.container) == 0 
                push!(agent, PostActStage(), env, action)
                push!(agent, PreActStage(), env)
                @test RLBase.plan!(agent, env) in (1,2)
                @test length(agent.trajectory.container) == 1

                #The following tests checks args / kwargs passed to policy cause an error
                @test_throws "MethodError: no method matching plan!(::Agent{RandomPolicy" RLBase.plan!(agent, env, 1)
                @test_throws "MethodError: no method matching plan!(::Agent{RandomPolicy" RLBase.plan!(agent, env, fake_kwarg = 1)
            end
        end
    end
end
