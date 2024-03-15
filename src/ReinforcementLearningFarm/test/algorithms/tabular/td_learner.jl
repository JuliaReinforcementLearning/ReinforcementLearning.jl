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

# Test Q! function
@testset "Q! function" begin
    approximator = TabularQApproximator(n_state=5, n_action=3)
    s = 1
    s_plus_one = 2
    a = 3
    α = 0.1
    π_ = 0.5
    γ = 0.9
    RLFarm.Q!(approximator, s, s_plus_one, a, α, π_, γ)
    @test RLFarm.Q(approximator, s, a) ≈ α * (π_ + γ * maximum(RLFarm.Q(approximator, s_plus_one)) - RLFarm.Q(approximator, s, a))
end

# Test optimise! function
@testset "optimise! function" begin
    approximator = TabularQApproximator(n_state=5, n_action=3)
    learner = TDLearner(approximator, :SARS)
    t = (state=1, next_state=2, action=3, reward=0.5, terminal=false)
    optimise!(learner, t)
    @test true  # Add your own assertion here
end
