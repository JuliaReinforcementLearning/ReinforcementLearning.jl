using Test
using Flux

@testset "Test TDLearner creation" begin
    approximator = TabularVApproximator(n_state=5)
    @test TDLearner(approximator, :SARS, γ=0.95, n=0) isa TDLearner

    approximator = TabularQApproximator(n_state=5, n_action=3)
    @test TDLearner(approximator, :SARS, γ=0.95, n=0) isa TDLearner
end

# Test TDLearner struct
@testset "TDLearner struct" begin
    approximator = TabularQApproximator(n_state=5, n_action=3)
    learner = TDLearner(approximator, :SARS)
    @test learner.approximator === approximator
    @test learner.γ == 1.0
    @test learner.n == 0
end

# Test bellman_update! function
@testset "bellman_update! function" begin
    learner = TDLearner(TabularQApproximator(n_state=5, n_action=3), :SARS)
    approximator = learner.approximator
    s = 1
    s_plus_one = 2
    a = 3
    α = learner.α
    π_ = 5.0
    γ = 0.9
    approximator.model[2, s_plus_one] = 15
    approximator.model[a, s] = 2

    # Following https://en.wikipedia.org/wiki/Q-learning#Algorithm
    q_should_be = (1-α) * RLCore.Q(approximator, s, a) + α * (π_ + γ * maximum(RLCore.Q(approximator, s_plus_one)))

    @test RLCore.bellman_update!(approximator, s, s_plus_one, a, π_, γ, α) ≈ q_should_be
    @test RLCore.Q(approximator, s, a) ≈ q_should_be
end

# Test optimise! function
@testset "optimise! function" begin
    learner = TDLearner(TabularQApproximator(n_state=5, n_action=3), :SARS)
    approximator = learner.approximator

    t = (state=1, next_state=2, action=3, reward=5.0, terminal=false)
    optimise!(learner, t)
    @test learner.approximator.model[t.action, t.state] ≈ 0.05
    optimise!(learner, t)
    @test learner.approximator.model[t.action, t.state] ≈ 0.0995

    for i in 1:1000
        optimise!(learner, t)
    end
    @test approximator.model[t.action, t.state] ≈ t.reward atol=0.01
end
