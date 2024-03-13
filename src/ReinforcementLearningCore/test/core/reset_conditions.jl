using Test
using ReinforcementLearningCore
@testset "Reset Conditions Tests" begin

    @testset "Test ResetIfEnvTerminated" begin
        policy = RandomPolicy()
        env = RandomWalk1D()
        reset_condition = ResetIfEnvTerminated()
        is_terminated(env) = true
        @test check!(reset_condition, policy, env) == true
    end

    @testset "Test ResetAfterNSteps" begin
        policy = RandomPolicy()
        env = RandomWalk1D()
        reset_condition = ResetAfterNSteps(3)
        @test check!(reset_condition, policy, env) == false
        @test check!(reset_condition, policy, env) == false
        @test check!(reset_condition, policy, env) == true
        @test check!(reset_condition, policy, env) == false
    end
