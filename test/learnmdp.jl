using ReinforcementLearningBase
using ReinforcementLearningEnvironmentDiscrete
using Random:seed!

@testset "learnmdp" begin

@testset "MDP" begin
    seed!(123)
    mdp = MDP(ns = 5, na = 3); 
    γ = .5
    mdpl = MDPLearner(mdp = mdp, γ = γ); policy_iteration!(mdpl)
    x = RLSetup(learner = QLearning(ns = 5, na = 3, γ = γ, λ = 0., α = 1e-3), 
                environment = mdp,
                stoppingcriterion = ConstantNumberSteps(10^6))
    learn!(x)
    @test mdpl.values ≈ getvalues(x.learner) atol=0.3
end

@testset "DetTreeMDP" begin
    seed!(123)
    mdp = DetTreeMDP()
    mdpl = MDPLearner(mdp = mdp, γ =.9); policy_iteration!(mdpl)
    x = RLSetup(learner = mdpl, policy = EpsilonGreedyPolicy(0., mdp.actionspace, s -> mdpl.policy[s]), 
                callbacks = [MeanReward()], environment = mdp, 
                stoppingcriterion = ConstantNumberEpisodes(2))
    run!(x)
    @test 5 * getvalue(x.callbacks[1]) ≈ maximum(mdp.reward[findall(x -> x != 0, 
                                                                    mdp.reward)])
end
end