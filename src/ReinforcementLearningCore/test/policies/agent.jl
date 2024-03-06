using ReinforcementLearningBase
import ReinforcementLearningCore.SRT

@testset "agent.jl" begin
    @testset "Agent Tests" begin
        a_1 = Agent(
            RandomPolicy(),
            Trajectory(
                CircularArraySARTSTraces(; capacity=1_000),
                DummySampler(),
            ),
        )
        a_2 = Agent(
            RandomPolicy(),
            Trajectory(
                CircularArraySARTSTraces(; capacity=1_000),
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
                @test action in (1, 2)
                @test length(agent.trajectory.container) == 0
                push!(agent, PostActStage(), env, action)
                push!(agent, PreActStage(), env)
                @test RLBase.plan!(agent, env) in (1, 2)
                @test length(agent.trajectory.container) == 1

                #The following tests checks args / kwargs passed to policy cause an error
                @test_throws "MethodError: no method matching plan!(::Agent{RandomPolicy" RLBase.plan!(agent, env, 1)
                @test_throws "MethodError: no method matching plan!(::Agent{RandomPolicy" RLBase.plan!(agent, env, fake_kwarg=1)
            end
        end
    end
    @testset "OfflineAgent" begin
        env = RandomWalk1D()
        a_1 = OfflineAgent(
            policy=RandomPolicy(),
            trajectory=Trajectory(
                CircularArraySARTSTraces(; capacity=1_000),
                DummySampler(),
            ),
        )
        @test a_1.offline_behavior.agent === nothing
        push!(a_1, PreExperimentStage(), env)
        @test isempty(a_1.trajectory.container)

        trajectory = Trajectory(
            CircularArraySARTSTraces(; capacity=1_000),
            DummySampler(),
        )

        a_2 = OfflineAgent(
            policy=RandomPolicy(),
            trajectory=trajectory,
            offline_behavior=OfflineBehavior(
                Agent(RandomPolicy(), trajectory),
                steps=5,
            )
        )
        push!(a_2, PreExperimentStage(), env)
        # We'll have 1 extra element where terminal is true 
        # if the environment was terminated mid-episode and restarted!
        ix = findfirst(x -> x.terminal, map(identity, a_2.trajectory.container))
        len = length(a_2.trajectory.container)
        max = isnothing(ix) || ix == len ? 5 : 6
        @test len == max

        for agent in [a_1, a_2]
            action = RLBase.plan!(agent, env)
            @test action in (1, 2)
            for stage in [PreEpisodeStage(), PreActStage(), PostActStage(), PostEpisodeStage()]
                push!(agent, stage, env)
                ix = findfirst(x -> x.terminal, map(identity, agent.trajectory.container))
                len = length(agent.trajectory.container)
                max = isnothing(ix) || ix == len ? 5 : 6
                @test len in (0, max)
            end
        end
    end
end
