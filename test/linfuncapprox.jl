import ReinforcementLearning: preprocessstate
struct OneHotPreprocessor 
    ns::Int64
end
preprocessstate(p::OneHotPreprocessor, s) = Float64[i == s for i in 1:p.ns]
for λ in [0, .8]
    mdp = MDP()
    x = RLSetup(learner = QLearning(λ = λ, tracekind = AccumulatingTraces),
                      # with ReplacingTraces the results will be different
                      # because of different replacingAccumulatingTraces 
                preprocessor = OneHotPreprocessor(mdp.observationspace.n),
                environment = mdp,
                stoppingcriterion = ConstantNumberSteps(100))
    seed!(124)
    s0 = mdp.state
    learn!(x)
    seed!(124)
    mdp.state = s0
    y = RLSetup(learner = QLearning(initvalue = 0., λ = λ,
                                    tracekind = AccumulatingTraces), 
                environment = mdp,
                stoppingcriterion = ConstantNumberSteps(100))
    learn!(y)
    @test x.learner.params ≈ y.learner.params 
end

for learner in [PolicyGradientBackward, EpisodicReinforce,
                ActorCriticPolicyGradient]
    mdp = MDP()
    x = RLSetup(learner = learner(),
                preprocessor = OneHotPreprocessor(mdp.observationspace.n), 
                environment = mdp,
                stoppingcriterion = ConstantNumberSteps(100))
    seed!(124)
    s0 = mdp.state
    learn!(x)
    seed!(124)
    mdp.state = s0
    y = RLSetup(learner = learner(initvalue = 0.), 
                environment = mdp,
                stoppingcriterion = ConstantNumberSteps(100))
    learn!(y)
    @test x.learner.params ≈ y.learner.params 
end

using Flux
struct Id end 
(l::Id)(x) = x
function testlinfuncapproxflux()
    ns = 10; na = 4;
    env = MDP(ns = ns, na = na, init = "deterministic")
    policy = ForcedPolicy(rand(1:na, 200))
    learner = DQN(Linear(ns, na), replaysize = 2, updatetargetevery = 1, 
                  updateevery = 1, startlearningat = 1, 
                  opttype = x -> Flux.SGD(x, .1/2), 
                  minibatchsize = 1, doubledqn = false)
    x = RLSetup(learner = learner, 
                preprocessor = OneHotPreprocessor(ns),
                policy = policy,
                environment = env, 
                callbacks = [EvaluationPerT(10^3, MeanReward()), RecordAll()],
                stoppingcriterion = ConstantNumberSteps(60))
    x2 = RLSetup(learner = QLearning(λ = 0, γ = .99, initvalue = 0., α = .1), 
                 policy = policy,
                 environment = env, 
                 callbacks = [EvaluationPerT(10^3, MeanReward()), RecordAll()], 
                 stoppingcriterion = ConstantNumberSteps(60))
    seed!(445)
    reset!(env)
    learn!(x)
    seed!(445)
    reset!(env)
    x2.policy.t = 1
    learn!(x2)
    @test x.learner.net.W.data ≈ x2.learner.params

    ns = 10; na = 4;
    env = MDP(ns = ns, na = na, init = "deterministic")
    policy = ForcedPolicy(rand(1:na, 200))
    learner = DeepActorCritic(net = Id(), nh = 10, na = 4, αcritic = 0.,
                              opttype = x -> Flux.SGD(x, .1), nsteps = 4)
    x = RLSetup(learner = learner, 
                preprocessor = OneHotPreprocessor(ns),
                policy = policy,
                environment = env, 
                callbacks = [EvaluationPerT(10^3, MeanReward()), RecordAll()],
                stoppingcriterion = ConstantNumberSteps(5))
    x2 = RLSetup(learner = ActorCriticPolicyGradient(nsteps = 4, αcritic = 0.), 
                 policy = policy,
                 environment = env, 
                 callbacks = [EvaluationPerT(10^3, MeanReward()), RecordAll()], 
                 stoppingcriterion = ConstantNumberSteps(5))
    seed!(445)
    reset!(env)
    learn!(x)
    seed!(445)
    reset!(env)
    x2.policy.t = 1
    learn!(x2)
    @test x.learner.policylayer.W.data ≈ x2.learner.params
end
testlinfuncapproxflux()
