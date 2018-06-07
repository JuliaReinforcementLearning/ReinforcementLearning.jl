# run this file with julia -p 4 to use 4 cores in the comparison
@everywhere begin
using ReinforcementLearning, Flux
loadenvironment("cartpole")
getenv() = (CartPole(), 4, 2)
# loadenvironment("mountaincar")
# getenv() = (MountainCar(maxsteps = 10^4), 2, 3)
function setup(learner, env, preprocessor = NoPreprocessor())
    cb = EvaluateGreedy(callback = EvaluationPerEpisode(TotalReward(),
                                                        returnmean = true),
                        stoppingcriterion = ConstantNumberEpisodes(200),
                        every = Episode(200))
    RLSetup(learner, env, ConstantNumberEpisodes(2000), 
            callbacks = [cb], preprocessor = preprocessor)
end
function acpg(i)
    env, ns, na = getenv()
    learner = ActorCriticPolicyGradient(na = na, ns = ns, α = .02,
                                        αcritic = 0.0, nsteps = 25)
    setup(learner, env)
end
function dqn(i)
    env, ns, na = getenv()
    learner = DQN(Chain(Dense(ns, 48, relu), Dense(48, 24, relu), Dense(24, na)),
                  updateevery = 1, updatetargetevery = 100,
                  startlearningat = 50, minibatchsize = 32,
                  doubledqn = false, replaysize = 10^3, 
                  opttype = x -> ADAM(x, .0005)) 
    setup(learner, env)
end
function tilingsarsa(i)
    env, ns, na = getenv()
    if typeof(env) <: CartPole
        high = [2.6, 4, .24, 3.4]
        low = -high
        nbins = 20 * ones(4)
        initvalue = 200
    elseif typeof(env) <: MountainCar
        high = [.5, .07]
        low = [-1.2, -.07]
        nbins = [16, 16]
        initvalue = 0
    end
    p0 = StateAggregator(low, high, nbins)
    preprocessor = TilingStateAggregator(p0, 8)
    learner = Sarsa(ns = 8*prod(nbins), na = na, λ = 0, α = .2/8, 
                    initvalue = initvalue)
    s = setup(learner, env, preprocessor)
    s.policy = EpsilonGreedyPolicy(0)
    s
end
rlsetupcreators = Dict("linear ACPG" => acpg, "DQN" => dqn, 
                       "tiling Sarsa" => tilingsarsa)
end # everywhere
@time res = compare(rlsetupcreators, 10, verbose = true)

using JLD2
@save tempname() * ".jld2" res

a = plotcomparison(res);
a["xlabel"] = "epochs";
a["ylabel"] = "average episode length greedy policy"
a
