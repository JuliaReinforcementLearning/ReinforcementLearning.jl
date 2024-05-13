using Test
using Flux
using ReinforcementLearningCore, ReinforcementLearningBase

# Mock explorer, environment, and learner
struct MockExplorer <: AbstractExplorer end
struct MockEnv <: AbstractEnv end
struct MockLearner <: AbstractLearner end

@testset "AbstractLearner Tests" begin
    @testset "Forward" begin

        function RLCore.forward(::MockLearner, state::Int)
            return [1.0, 2.0]
        end

        RLBase.state(::MockEnv, ::Observation, ::DefaultPlayer) = 1
        RLBase.state(::MockEnv, ::Observation{Any}, ::Player) = 1

        env = MockEnv()
        learner = MockLearner()

        output = RLCore.forward(learner, env)
        @test output == Float64[1.0, 2.0]

        output = RLCore.forward(learner, env, Player(1))
        @test output == Float64[1.0, 2.0]
    end

    @testset "Plan" begin
        function RLBase.plan!(::MockExplorer, learner::MockLearner, env::MockEnv)
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
        function RLBase.action_space(::MockEnv, ::Player)
            return [1, 2]
        end

        function RLBase.plan!(::MockExplorer, learner::MockLearner, env::MockEnv, p::Player)
            return rand(2)
        end

        env = MockEnv()
        learner = MockLearner()
        explorer = MockExplorer()
        player = Player(:player1)

        output = RLBase.plan!(explorer, learner, env, player)

        @test typeof(output) == Array{Float64,1}
        @test length(output) == 2
    end

    @testset "optimise!" begin
        tr = Trajectory(
                    CircularArraySARTSTraces(; capacity = 1_000),
                    BatchSampler(1),
                    InsertSampleRatioController(n_inserted = -1),
                )
        @test optimise!(MockLearner(), PreActStage(), tr) == nothing
    end
end
