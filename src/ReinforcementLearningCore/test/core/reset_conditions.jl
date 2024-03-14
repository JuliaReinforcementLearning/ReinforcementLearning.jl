using Test
using ReinforcementLearningCore

@testset "Reset Conditions Tests" begin

    @testset "Test ResetIfEnvTerminated" begin
        policy = RandomPolicy()
        env = RandomWalk1D()
        reset_condition = ResetIfEnvTerminated()
        env.pos = 1
        @test RLCore.check!(reset_condition, policy, env) == true
    end

    @testset "Test ResetAfterNSteps" begin
        policy = RandomPolicy()
        env = RandomWalk1D()
        reset_condition = ResetAfterNSteps(3)
        @test RLCore.check!(reset_condition, policy, env) == false
        @test RLCore.check!(reset_condition, policy, env) == false
        @test RLCore.check!(reset_condition, policy, env) == false
        @test RLCore.check!(reset_condition, policy, env) == true
        @test RLCore.check!(reset_condition, policy, env) == false
    end
end
