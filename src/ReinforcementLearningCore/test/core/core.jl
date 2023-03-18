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
                RandomPolicy(),
                Trajectory(
                    CircularArraySARTTraces(; capacity = 1_000),
                    BatchSampler(1),
                    InsertSampleRatioController(n_inserted = -1),
                ),
                cache=SRT{Int, Float64, Bool}(),
            )
            stop_condition = StopAfterStep(123; is_show_progress=false)
            hook = StepsPerEpisode()
            run(agent, env, stop_condition, hook)
            @test sum(hook[]) == length(agent.trajectory.container)
        end        
    end
end
