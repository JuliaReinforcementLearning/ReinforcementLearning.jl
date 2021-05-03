@testset "CommonRLInterface" begin
@testset "MDPEnv" begin
    struct RLTestMDP <: MDP{Int, Int} end

    POMDPs.actions(m::RLTestMDP) = [-1, 1]
    POMDPs.transition(m::RLTestMDP, s, a) = Deterministic(clamp(s + a, 1, 3))
    POMDPs.initialstate(m::RLTestMDP) = Deterministic(1)
    POMDPs.isterminal(m::RLTestMDP, s) = s == 3
    POMDPs.reward(m::RLTestMDP, s, a, sp) = sp
    POMDPs.states(m::RLTestMDP) = 1:3

    env = convert(RLBase.AbstractEnv, convert(CRL.AbstractEnv, RLTestMDP()))
    RLBase.test_runnable!(env)
end

@testset "POMDPEnv" begin

    struct RLTestPOMDP <: POMDP{Int, Int, Int} end

    POMDPs.actions(m::RLTestPOMDP) = [-1, 1]
    POMDPs.states(m::RLTestPOMDP) = 1:3
    POMDPs.transition(m::RLTestPOMDP, s, a) = Deterministic(clamp(s + a, 1, 3))
    POMDPs.observation(m::RLTestPOMDP, s, a, sp) = Deterministic(sp + 1)
    POMDPs.initialstate(m::RLTestPOMDP) = Deterministic(1)
    POMDPs.initialobs(m::RLTestPOMDP, s) = Deterministic(s + 1)
    POMDPs.isterminal(m::RLTestPOMDP, s) = s == 3
    POMDPs.reward(m::RLTestPOMDP, s, a, sp) = sp
    POMDPs.observations(m::RLTestPOMDP) = 2:4

    env = convert(RLBase.AbstractEnv, convert(CRL.AbstractEnv, RLTestPOMDP()))

    RLBase.test_runnable!(env)
end
end