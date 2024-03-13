using Test
using ReinforcementLearningCore

# Test ResetAtTerminal
function test_reset_at_terminal()
    policy = nothing
    env = nothing
    reset_condition = ResetAtTerminal()
    @test reset_condition(policy, env) == is_terminated(env)
end

# Test ResetAfterNSteps
function test_reset_after_n_steps()
    policy = nothing
    env = nothing
    reset_condition = ResetAfterNSteps(5)
    for i in 1:10
        @test reset_condition(policy, env) == (i % 5 == 0)
    end
end

# Run tests
@testset "Reset Conditions" begin
    test_reset_at_terminal()
    test_reset_after_n_steps()
end
