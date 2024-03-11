using Test
using Flux

@testset "AbstractLearner Tests" begin
    @testset "Forward" begin
        # Mock environment and learner
        struct MockEnv <: AbstractEnv end
        struct MockLearner <: AbstractLearner end

        function RLCore.forward(::MockLearner, ::AbstractState)
            return rand(2)
        end

        env = MockEnv()
        learner = MockLearner()

        output = forward(learner, env)

        @test typeof(output) == Array{Float64,1}
        @test length(output) == 2
    end

    @testset "Plan" begin
        # Mock explorer, environment, and learner
        struct MockExplorer <: AbstractExplorer end
        struct MockEnv <: AbstractEnv end
        struct MockLearner <: AbstractLearner end

        function RLBase.plan!(::MockExplorer, ::AbstractState, ::AbstractActionSpace)
            return rand(2)
        end

        env = MockEnv()
        learner = MockLearner()
        explorer = MockExplorer()

        output = RLBase.plan!(explorer, learner, env)

        @test typeof(output) == Array{Float64,1}
        @test length(output) == 2
    end

    @testset "Plan with Player" begin
        # Mock explorer, environment, and learner
        struct MockExplorer <: AbstractExplorer end
        struct MockEnv <: AbstractEnv end
        struct MockLearner <: AbstractLearner end

        function RLBase.plan!(::MockExplorer, ::AbstractState, ::AbstractActionSpace)
            return rand(2)
        end

        env = MockEnv()
        learner = MockLearner()
        explorer = MockExplorer()
        player = :player1

        output = RLBase.plan!(explorer, learner, env, player)

        @test typeof(output) == Array{Float64,1}
        @test length(output) == 2
    end

    @testset "optimise!" begin
        struct MockLearner <: AbstractLearner end
        tr = Trajectory(
                    CircularArraySARTSTraces(; capacity = 1_000),
                    BatchSampler(1),
                    InsertSampleRatioController(n_inserted = -1),
                )
        @test optimise!(MockLearner(), PreActStage(), tr) is nothing
    end
end
