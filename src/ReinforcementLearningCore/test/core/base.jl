using ReinforcementLearningBase
using TimerOutputs

@testset "core" begin
    @testset "simple workflow" begin
        @testset "StopAfterNSteps" begin
            agent = Agent(
                RandomPolicy(),
                Trajectory(
                    CircularArraySARTSTraces(; capacity = 1_000),
                    BatchSampler(1),
                    InsertSampleRatioController(n_inserted = -1),
                ),
            )
            env = RandomWalk1D()
            stop_condition = StopAfterNSteps(123)
            hook = StepsPerEpisode()
            run(agent, env, stop_condition, hook)

            @test sum(hook[]) + length(hook[]) - 1 == length(agent.trajectory.container)
        end

        @testset "StopAfterNEpisodes" begin
            agent = Agent(
                RandomPolicy(),
                Trajectory(
                    CircularArraySARTSTraces(; capacity = 1_000),
                    BatchSampler(1),
                    InsertSampleRatioController(n_inserted = -1),
                ),
            )
            env = RandomWalk1D()
            stop_condition = StopAfterNEpisodes(10)
            hook = StepsPerEpisode()
            run(agent, env, stop_condition, hook)

            @test length(hook[]) == 10
        end      
    end

    @testset "Debug Timer" begin
        RLCore.TimerOutputs.enable_debug_timings(RLCore)

        env = RandomWalk1D()
        agent = Agent(
            RandomPolicy(legal_action_space(env)),
            Trajectory(
                CircularArraySARTSTraces(; capacity = 1_000),
                BatchSampler(1),
                InsertSampleRatioController(n_inserted = -1),
            )
        )            
        stop_condition = StopAfterNSteps(123; is_show_progress=false)
        hook = StepsPerEpisode()
        run(agent, env, stop_condition, hook)
        @test RLCore.timer isa TimerOutputs.TimerOutput
    end

    @testset "Experiment" begin
        # Create an instance of Experiment
        policy = Agent(
            RandomPolicy(),
            Trajectory(
                CircularArraySARTSTraces(; capacity = 1_000),
                BatchSampler(1),
                InsertSampleRatioController(n_inserted = -1),
            ),
        )
        env = RandomWalk1D()
        stop_condition = StopAfterNEpisodes(10)
        hook = StepsPerEpisode()

        exp = Experiment(policy, env, stop_condition, hook)

        # Test that the fields are correctly assigned
        @test exp.policy === policy
        @test exp.env === env
        @test exp.stop_condition === stop_condition
        @test exp.hook === hook

        # Test that the Experiment is callable
        run(exp)
        @test length(hook[]) == 10
    end

end
