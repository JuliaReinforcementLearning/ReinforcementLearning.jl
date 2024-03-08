using Test
using Flux
using ReinforcementLearningCore
using ReinforcementLearningZoo
using Test


@testset "Test TDLearner creation" begin
    approximator = TabularVApproximator(n_state=5)
    @test TDLearner(approximator, :SARS, γ=0.95, n=0) isa TDLearner

    approximator = TabularQApproximator(n_state=5, n_action=3)
    @test TDLearner(approximator, :SARS, γ=0.95, n=0) isa TDLearner
end
