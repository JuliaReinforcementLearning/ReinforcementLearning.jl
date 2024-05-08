using ReinforcementLearningBase

struct TestEnv <: RLBase.AbstractEnv
    state::Int
end

function RLBase.state(env::TestEnv, ::Observation, ::DefaultPlayer)
    return env.state
end

@testset "MultiAgent" begin
    @test MultiAgent(2) isa MultiAgent
    @test_throws ArgumentError MultiAgent(1) 
    @test_throws ArgumentError MultiAgent(-1)
end

@testset "InformationSet" begin
    InformationSet() isa RLBase.AbstractStateStyle
end

@testset "InternalState" begin
    InternalState() isa RLBase.AbstractStateStyle
end

@testset "Observation" begin
    Observation() isa RLBase.AbstractStateStyle
end

@testset "EpisodeStyle" begin
    EpisodeStyle(TestEnv(10)) isa RLBase.AbstractEpisodeStyle
end

@testset "AbstractEnv" begin
    @test TestEnv(10) isa RLBase.AbstractEnv
    @test TestEnv(10) == TestEnv(10)
    @test Base.hash(TestEnv(10), UInt64(0)) == Base.hash(TestEnv(10), UInt64(0))
end

@testset "players" begin
    @test simultaneous_player(TestEnv(10)) == SimultaneousPlayer()
    @test RLBase.players(TestEnv(10)) == (DefaultPlayer(),)
end
