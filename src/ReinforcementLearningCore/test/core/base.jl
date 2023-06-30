using ReinforcementLearningCore: SRT
using ReinforcementLearningBase

@testset "core" begin
    @testset "simple workflow" begin
        @testset "StopAfterStep" begin
            agent = Agent(
                RandomPolicy(),
                Trajectory(
                    CircularArraySARTTraces(; capacity = 1_000),
                    BatchSampler(1),
                    InsertSampleRatioController(n_inserted = -1),
                ),
            )
            env = RandomWalk1D()
            stop_condition = StopAfterStep(123)
            hook = StepsPerEpisode()
            run(agent, env, stop_condition, hook)

            @test sum(hook[]) == length(agent.trajectory.container)
        end

        @testset "StopAfterEpisode" begin
            agent = Agent(
                RandomPolicy(),
                Trajectory(
                    CircularArraySARTTraces(; capacity = 1_000),
                    BatchSampler(1),
                    InsertSampleRatioController(n_inserted = -1),
                ),
            )
            env = RandomWalk1D()
            stop_condition = StopAfterEpisode(10)
            hook = StepsPerEpisode()
            run(agent, env, stop_condition, hook)

            @test sum(hook[]) == length(agent.trajectory.container)
        end

        @testset "StopAfterStep, use type stable Agent" begin
            env = RandomWalk1D()
            agent = Agent(
                RandomPolicy(legal_action_space(env)),
                Trajectory(
                    CircularArraySARTTraces(; capacity = 1_000),
                    BatchSampler(1),
                    InsertSampleRatioController(n_inserted = -1),
                ),
                SRT{Any, Any, Any}(),
            )            
            stop_condition = StopAfterStep(123; is_show_progress=false)
            hook = StepsPerEpisode()
            run(agent, env, stop_condition, hook)
            @test sum(hook[]) == length(agent.trajectory.container)
        end        
    end

    @testset "Debug Timer" begin
        RLCore.TimerOutputs.enable_debug_timings(RLCore)

        env = RandomWalk1D()
        agent = Agent(
            RandomPolicy(legal_action_space(env)),
            Trajectory(
                CircularArraySARTTraces(; capacity = 1_000),
                BatchSampler(1),
                InsertSampleRatioController(n_inserted = -1),
            ),
            SRT{Any, Any, Any}(),
        )            
        stop_condition = StopAfterStep(123; is_show_progress=false)
        hook = StepsPerEpisode()
        run(agent, env, stop_condition, hook)
        @test RLCore.timer isa TimerOutputs.TimerOutput
    end
end
