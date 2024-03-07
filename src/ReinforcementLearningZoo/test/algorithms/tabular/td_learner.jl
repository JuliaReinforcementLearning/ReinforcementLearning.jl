using ReinforcementLearningCore
using ReinforcementLearningZoo
using Test

@testset "Test TDLearner creation" begin
    approximator = TabularVApproximator(n_state=5)
    TDLearner(approximator, :SARS, γ=0.95, n=0)

    approximator = TabularQApproximator(n_state=5, n_action=3)
    TDLearner(approximator, :SARS, γ=0.95, n=0)
end
