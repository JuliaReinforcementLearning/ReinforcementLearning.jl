
using Test
using ReinforcementLearningCore
using Flux

@testset "Constructors" begin
    @test TabularApproximator(fill(1, 10, 10), InvDecay(0.5)) isa TabularApproximator
    @test TabularVApproximator(n_state = 10, opt = InvDecay(0.5)) isa
          TabularApproximator{Vector{Float64},InvDecay}
    @test TabularQApproximator(n_state = 10, n_action = 10, opt = InvDecay(0.5)) isa
          TabularApproximator{Matrix{Float64},InvDecay}
end

@testset "RLCore.forward" begin
    v_approx = TabularVApproximator(n_state = 10, opt = InvDecay(0.5))
    @test RLCore.forward(v_approx, 1) == 0.0

    env = RockPaperScissorsEnv()
    @test RLCore.forward(v_approx, env) == 0.0

    q_approx = TabularQApproximator(n_state = 5, n_action = 10, opt = InvDecay(0.5))
    @test RLCore.forward(q_approx, 1) == zeros(Float64, 10)
    @test RLCore.forward(q_approx, 1, 5) == 0.0
    @test RLCore.forward(q_approx, env) == zeros(10)
end
