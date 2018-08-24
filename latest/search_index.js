var documenterSearchIndex = {"docs": [

{
    "location": "#",
    "page": "Introduction",
    "title": "Introduction",
    "category": "page",
    "text": ""
},

{
    "location": "#ReinforcementLearning-1",
    "page": "Introduction",
    "title": "ReinforcementLearning",
    "category": "section",
    "text": "(Image: Documentation) (Image: Build Status) (Image: codecov)A reinforcement learning package for Julia."
},

{
    "location": "#What-is-reinforcement-learning?-1",
    "page": "Introduction",
    "title": "What is reinforcement learning?",
    "category": "section",
    "text": "Wikipedia\nNew Sutton & Barto book"
},

{
    "location": "#Features-1",
    "page": "Introduction",
    "title": "Features",
    "category": "section",
    "text": ""
},

{
    "location": "#Learning-methods-1",
    "page": "Introduction",
    "title": "Learning methods",
    "category": "section",
    "text": "name discrete states linear approximation non-linear approximation\nQ-learning/SARSA(λ) ✓ ✓ \nn-step Q-learning/SARSA ✓ ✓ \nOnline Policy Gradient ✓ ✓ \nEpisodic Reinforce ✓ ✓ \nn-step Actor-Critic Policy-Gradient ✓ ✓ ✓\nMonte Carlo Control ✓  \nPrioritized Sweeping ✓  \n(double) DQN  ✓ ✓"
},

{
    "location": "#Environments-1",
    "page": "Introduction",
    "title": "Environments",
    "category": "section",
    "text": "name state space action space\nCartpole 4D discrete\nMountain Car 2D discrete\nPendulum 3D 1D\nAtari pixel images discrete\nVizDoom pixel images discrete\nPOMDPs, MDPs, Mazes, Cliffwalking discrete discrete\nOpenAi Gym (using PyCall) see here see here"
},

{
    "location": "#Preprocessors-1",
    "page": "Introduction",
    "title": "Preprocessors",
    "category": "section",
    "text": "State Aggregation\nTile Coding\nRandom Projections\nRadial Basis Functions"
},

{
    "location": "#Helper-Functions-1",
    "page": "Introduction",
    "title": "Helper Functions",
    "category": "section",
    "text": "comparison of different methods\ncallbacks to track performance, change exploration policy, save models during learning etc."
},

{
    "location": "#Installation-1",
    "page": "Introduction",
    "title": "Installation",
    "category": "section",
    "text": "(v1.0) pkg> add ReinforcementLearningor in julia v0.6Pkg.add(\"ReinforcementLearning\")"
},

{
    "location": "#Credits-1",
    "page": "Introduction",
    "title": "Credits",
    "category": "section",
    "text": "Main author: Johanni Brea\nContributions: Marco Lehmann, Raphaël Nunes"
},

{
    "location": "#Contribute-1",
    "page": "Introduction",
    "title": "Contribute",
    "category": "section",
    "text": "Contributions are highly welcome. Please have a look at the issues."
},

{
    "location": "usage/#",
    "page": "Usage",
    "title": "Usage",
    "category": "page",
    "text": ""
},

{
    "location": "usage/#Simple-usage-1",
    "page": "Usage",
    "title": "Simple usage",
    "category": "section",
    "text": "Choose a learner.\nChoose an environment.\nChoose a stopping criterion.\n(Optionally) choose callbacks.\n(Optionally) choose a preprocessor.\nDefine an RLSetup.\nLearn with learn!.\nLook at results with getvalue."
},

{
    "location": "usage/#Example-1-1",
    "page": "Usage",
    "title": "Example 1",
    "category": "section",
    "text": "using ReinforcementLearning\n\nlearner = QLearning()\nenv = MDP()\nstop = ConstantNumberSteps(10^3)\nx = RLSetup(learner, env, stop, callbacks = [TotalReward()])\nlearn!(x)\ngetvalue(x.callbacks[1])"
},

{
    "location": "usage/#Example-2-1",
    "page": "Usage",
    "title": "Example 2",
    "category": "section",
    "text": "using ReinforcementLearning, Flux\n\nlearner = DQN(Chain(Dense(4, 24, relu), Dense(24, 48, relu), Dense(48, 2)),\n              opttype = x -> ADAM(x, .001))\nloadenvironment(\"cartpole\")\nenv = CartPole()\nstop = ConstantNumberEpisodes(2*10^3)\ncallbacks = [EvaluateGreedy(EvaluationPerEpisode(TimeSteps(), returnmean=true),\n                            ConstantNumberEpisodes(100), every = Episode(100)),\n             EvaluationPerEpisode(TimeSteps()),\n             Progress()]\nx = RLSetup(learner, env, stop, callbacks = callbacks)\nlearn!(x)\ngetvalue(x.callbacks[1])"
},

{
    "location": "usage/#Comparisons-1",
    "page": "Usage",
    "title": "Comparisons",
    "category": "section",
    "text": "See section Comparison."
},

{
    "location": "usage/#Examples-1",
    "page": "Usage",
    "title": "Examples",
    "category": "section",
    "text": "See environments"
},

{
    "location": "comparison/#",
    "page": "Comparison",
    "title": "Comparison",
    "category": "page",
    "text": ""
},

{
    "location": "comparison/#ReinforcementLearning.compare-Tuple{Any,Any}",
    "page": "Comparison",
    "title": "ReinforcementLearning.compare",
    "category": "method",
    "text": "compare(rlsetupcreators::Dict, N; callbackid = 1, verbose = false)\n\nRun different setups in dictionary rlsetupcreators N times. The dictionary has elements \"name\" => createrlsetup, where createrlsetup is a function that has a single integer argument (id of the comparison; useful for saving  intermediate results). For each run, getvalue(rlsetup.callbacks[callbackid]) gets entered as result in a DataFrame with columns \"name\", \"result\", \"seed\".\n\n\n\n"
},

{
    "location": "comparison/#ReinforcementLearning.plotcomparison-Tuple{Any}",
    "page": "Comparison",
    "title": "ReinforcementLearning.plotcomparison",
    "category": "method",
    "text": "plotcomparison(df; nmaxpergroup = 20, linestyles = [], \n                   showbest = true, axisoptions = @pgf {})\n\nPlots results obtained with compare using PGFPlotsX.\n\n\n\n"
},

{
    "location": "comparison/#comparison-1",
    "page": "Comparison",
    "title": "Comparison Tools",
    "category": "section",
    "text": "Modules = [ReinforcementLearning]\nPages   = [\"compare.jl\"]"
},

{
    "location": "learning/#ReinforcementLearning.RLSetup",
    "page": "Learning",
    "title": "ReinforcementLearning.RLSetup",
    "category": "type",
    "text": "@with_kw mutable struct RLSetup{Tl,Tb,Tp,Tpp,Te,Ts}\n    learner::Tl\n    environment::Te\n    stoppingcriterion::Ts\n    preprocessor::Tpp = NoPreprocessor()\n    buffer::Tb = defaultbuffer(learner, environment, preprocessor)\n    policy::Tp = defaultpolicy(learner, buffer)\n    callbacks::Array{Any, 1} = []\n    islearning::Bool = true\n    fillbuffer::Bool = islearning\n\n\n\n"
},

{
    "location": "learning/#ReinforcementLearning.RLSetup-Tuple{Any,Any,Any}",
    "page": "Learning",
    "title": "ReinforcementLearning.RLSetup",
    "category": "method",
    "text": "RLSetup(learner, env, stop; kargs...) = RLSetup(learner = learner,\n                                                environment = env,\n                                                stoppingcriterion = stop;\n                                                kargs...)\n\n\n\n"
},

{
    "location": "learning/#ReinforcementLearning.learn!-Tuple{Any}",
    "page": "Learning",
    "title": "ReinforcementLearning.learn!",
    "category": "method",
    "text": "learn!(rlsetup)\n\nRuns an rlsetup with learning.\n\n\n\n"
},

{
    "location": "learning/#ReinforcementLearning.run!-Tuple{Any}",
    "page": "Learning",
    "title": "ReinforcementLearning.run!",
    "category": "method",
    "text": "run!(rlsetup)\n\nRuns an rlsetup without learning.\n\n\n\n"
},

{
    "location": "learning/#",
    "page": "Learning",
    "title": "Learning",
    "category": "page",
    "text": "Modules = [ReinforcementLearning]\nPages   = [\"rlsetup.jl\", \"learn.jl\"]"
},

{
    "location": "learners/#",
    "page": "Learners",
    "title": "Learners",
    "category": "page",
    "text": ""
},

{
    "location": "learners/#learners-1",
    "page": "Learners",
    "title": "Learners",
    "category": "section",
    "text": ""
},

{
    "location": "learners/#ReinforcementLearning.AccumulatingTraces",
    "page": "Learners",
    "title": "ReinforcementLearning.AccumulatingTraces",
    "category": "type",
    "text": "struct AccumulatingTraces <: AbstractTraces\n    λ::Float64\n    γλ::Float64\n    trace::Array{Float64, 2}\n    minimaltracevalue::Float64\n\nDecaying traces with factor γλ. \n\nTraces are updated according to e(a s)   1 + e(a s) for the current action-state pair and e(a s)    e(a s) for all other pairs unless e(a s)  minimaltracevalue where the trace is set to 0  (for computational efficiency).\n\n\n\n"
},

{
    "location": "learners/#ReinforcementLearning.AccumulatingTraces-Tuple{}",
    "page": "Learners",
    "title": "ReinforcementLearning.AccumulatingTraces",
    "category": "method",
    "text": "AccumulatingTraces(ns, na, λ::Float64, γ::Float64; minimaltracevalue = 1e-12)\n\n\n\n"
},

{
    "location": "learners/#ReinforcementLearning.NoTraces",
    "page": "Learners",
    "title": "ReinforcementLearning.NoTraces",
    "category": "type",
    "text": "struct NoTraces <: AbstractTraces\n\nNo eligibility traces, i.e. e(a s) = 1 for current action a and state s and zero otherwise.\n\n\n\n"
},

{
    "location": "learners/#ReinforcementLearning.ReplacingTraces",
    "page": "Learners",
    "title": "ReinforcementLearning.ReplacingTraces",
    "category": "type",
    "text": "struct ReplacingTraces <: AbstractTraces\n    λ::Float64\n    γλ::Float64\n    trace::Array{Float64, 2}\n    minimaltracevalue::Float64\n\nDecaying traces with factor γλ. \n\nTraces are updated according to e(a s)   1 for the current action-state pair and e(a s)    e(a s) for all other pairs unless e(a s)  minimaltracevalue where the trace is set to 0  (for computational efficiency).\n\n\n\n"
},

{
    "location": "learners/#ReinforcementLearning.ReplacingTraces-Tuple{}",
    "page": "Learners",
    "title": "ReinforcementLearning.ReplacingTraces",
    "category": "method",
    "text": "ReplacingTraces(ns, na, λ::Float64, γ::Float64; minimaltracevalue = 1e-12)\n\n\n\n"
},

{
    "location": "learners/#TD-Learner-1",
    "page": "Learners",
    "title": "TD Learner",
    "category": "section",
    "text": "Modules = [ReinforcementLearning]\nPages   = [\"tdlearning.jl\", \"traces.jl\"]"
},

{
    "location": "learners/#initunseen-1",
    "page": "Learners",
    "title": "Initial values, novel actions and unseen values",
    "category": "section",
    "text": "For td-error dependent methods, the exploration-exploitation trade-off depends on the initvalue and the unseenvalue.  To distinguish actions that were never choosen before, i.e. novel actions, the default initial Q-value (field param) is initvalue = Inf64. In a state with novel actions, the policy determines how to deal with novel actions. To compute the td-error the unseenvalue is used for states with novel actions.  One way to achieve agressively exploratory behavior is to assure that unseenvalue (or initvalue) is larger than the largest possible Q-value."
},

{
    "location": "learners/#ReinforcementLearning.Critic",
    "page": "Learners",
    "title": "ReinforcementLearning.Critic",
    "category": "type",
    "text": "mutable struct Critic <: AbstractBiasCorrector\n    α::Float64\n    V::Array{Float64, 1}\n\n\n\n"
},

{
    "location": "learners/#ReinforcementLearning.Critic-Tuple{}",
    "page": "Learners",
    "title": "ReinforcementLearning.Critic",
    "category": "method",
    "text": "Critic(; γ = .9, α = .1, ns = 10, initvalue = 0.)\n\n\n\n"
},

{
    "location": "learners/#ReinforcementLearning.NoBiasCorrector",
    "page": "Learners",
    "title": "ReinforcementLearning.NoBiasCorrector",
    "category": "type",
    "text": "struct NoBiasCorrector <: AbstractBiasCorrector\n\n\n\n"
},

{
    "location": "learners/#ReinforcementLearning.RewardLowpassFilterBiasCorrector",
    "page": "Learners",
    "title": "ReinforcementLearning.RewardLowpassFilterBiasCorrector",
    "category": "type",
    "text": "mutable struct RewardLowpassFilterBiasCorrector <: AbstractBiasCorrector\nλ::Float64\nrmean::Float64\n\nFilters the reward with factor λ and uses effective reward (r - rmean) to update the parameters.\n\n\n\n"
},

{
    "location": "learners/#ReinforcementLearning.ActorCriticPolicyGradient-Tuple{}",
    "page": "Learners",
    "title": "ReinforcementLearning.ActorCriticPolicyGradient",
    "category": "method",
    "text": "ActorCriticPolicyGradient(; nsteps = 1, γ = .9, ns = 10, na = 4, \n                            α = .1, αcritic = .1, initvalue = Inf64)\n\n\n\n"
},

{
    "location": "learners/#ReinforcementLearning.EpisodicReinforce-Tuple{}",
    "page": "Learners",
    "title": "ReinforcementLearning.EpisodicReinforce",
    "category": "method",
    "text": "EpisodicReinforce(; kwargs...) = PolicyGradientForward(; kwargs...)\n\n\n\n"
},

{
    "location": "learners/#ReinforcementLearning.AbstractPolicyGradient",
    "page": "Learners",
    "title": "ReinforcementLearning.AbstractPolicyGradient",
    "category": "type",
    "text": "mutable struct PolicyGradientBackward <: AbstractPolicyGradient\n    ns::Int64 = 10\n    na::Int64 = 4\n    γ::Float64 = .9\n    α::Float64 = .1\n    initvalue::Float64 = 0.\n    params::Array{Float64, 2} = zeros(na, ns) + initvalue\n    traces::AccumulatingTraces = AccumulatingTraces(ns, na, 1., γ, \n                                                    trace = zeros(na, ns))\n    biascorrector::T = NoBiasCorrector()\n\nPolicy gradient learning in the backward view.\n\nThe parameters are updated according to paramsa s +=  * r_eff * ea s where r_eff =  r for NoBiasCorrector, r_eff =  r - rmean for RewardLowpassFilterBiasCorrector and e[a, s] is the eligibility trace.\n\n\n\n"
},

{
    "location": "learners/#ReinforcementLearning.PolicyGradientForward",
    "page": "Learners",
    "title": "ReinforcementLearning.PolicyGradientForward",
    "category": "type",
    "text": "mutable struct PolicyGradientForward <: AbstractPolicyGradient\n    ns::Int64 = 10\n    na::Int64 = 4\n    γ::Float64 = .9\n    α::Float64 = .1\n    initvalue::Float64 = 0.\n    params::Array{Float64, 2} = zeros(na, ns) + initvalue\n    biascorrector::Tb = NoBiasCorrector()\n    nsteps::Int64 = typemax(Int64)\n\n\n\n"
},

{
    "location": "learners/#Policy-Gradient-Learner-1",
    "page": "Learners",
    "title": "Policy Gradient Learner",
    "category": "section",
    "text": "Modules = [ReinforcementLearning]\nPages   = [\"policygradientlearning.jl\"]"
},

{
    "location": "learners/#ReinforcementLearning.MonteCarlo",
    "page": "Learners",
    "title": "ReinforcementLearning.MonteCarlo",
    "category": "type",
    "text": "mutable struct MonteCarlo <: AbstractReinforcementLearner\n    ns::Int64 = 10\n    na::Int64 = 4\n    γ::Float64 = .9\n    initvalue = 0.\n    Nsa::Array{Int64, 2} = zeros(Int64, na, ns)\n    Q::Array{Float64, 2} = zeros(na, ns) + initvalue\n\nEstimate Q values by averaging over returns.\n\n\n\n"
},

{
    "location": "learners/#N-step-Learner-1",
    "page": "Learners",
    "title": "N-step Learner",
    "category": "section",
    "text": "Modules = [ReinforcementLearning]\nPages   = [\"montecarlo.jl\"]"
},

{
    "location": "learners/#ReinforcementLearning.MDPLearner",
    "page": "Learners",
    "title": "ReinforcementLearning.MDPLearner",
    "category": "type",
    "text": "@with_kw struct MDPLearner\n    mdp::MDP = MDP()\n    γ::Float64 = .9\n    policy::Array{Int64, 1} = ones(Int64, mdp.ns)\n    values::Array{Float64, 1} = zeros(mdp.ns)\n\nUsed to solve mdp with discount factor γ.\n\n\n\n"
},

{
    "location": "learners/#ReinforcementLearning.policy_iteration!-Tuple{ReinforcementLearning.MDPLearner}",
    "page": "Learners",
    "title": "ReinforcementLearning.policy_iteration!",
    "category": "method",
    "text": "policy_iteration!(mdplearner::MDPLearner)\n\nSolve MDP with policy iteration using MDPLearner.\n\n\n\n"
},

{
    "location": "learners/#ReinforcementLearning.SmallBackups",
    "page": "Learners",
    "title": "ReinforcementLearning.SmallBackups",
    "category": "type",
    "text": "mutable struct SmallBackups <: AbstractReinforcementLearner\n    ns::Int64 = 10\n    na::Int64 = 4\n    γ::Float64 = .9\n    initvalue::Float64 = Inf64\n    maxcount::UInt64 = 3\n    minpriority::Float64 = 1e-8\n    M::Int64 = 1\n    counter::Int64 = 0\n    Q::Array{Float64, 2} = zeros(na, ns) .+ initvalue\n    V::Array{Float64, 1} = zeros(ns) .+ (initvalue == Inf64 ? 0. : initvalue)\n    U::Array{Float64, 1} = zeros(ns) .+ (initvalue == Inf64 ? 0. : initvalue)\n    Nsa::Array{Int64, 2} = zeros(Int64, na, ns)\n    Ns1a0s0::Array{Dict{Tuple{Int64, Int64}, Int64}, 1} = [Dict{Tuple{Int64, Int64}, Int64}() for _ in 1:ns]\n    queue::PriorityQueue = PriorityQueue(Base.Order.Reverse, zip(Int64[], Float64[]))\n\nSee Harm Van Seijen, Rich Sutton ; Proceedings of the 30th International Conference on Machine Learning, PMLR 28(3):361-369, 2013.\n\nmaxcount defines the maximal number of backups per action, minpriority is the smallest priority still added to the queue.\n\n\n\n"
},

{
    "location": "learners/#Model-Based-Learner-1",
    "page": "Learners",
    "title": "Model Based Learner",
    "category": "section",
    "text": "Modules = [ReinforcementLearning]\nPages   = [\"mdplearner.jl\", \"prioritizedsweeping.jl\"]"
},

{
    "location": "learners/#ReinforcementLearning.DQN",
    "page": "Learners",
    "title": "ReinforcementLearning.DQN",
    "category": "type",
    "text": "mutable struct DQN{Tnet,TnetT,ToptT,Topt}\n    γ::Float64 = .99\n    na::Int64\n    net::TnetT\n    targetnet::Tnet = Flux.mapleaves(Flux.Tracker.data, deepcopy(net))\n    policynet::Tnet = Flux.mapleaves(Flux.Tracker.data, net)\n    updatetargetevery::Int64 = 500\n    t::Int64 = 0\n    updateevery::Int64 = 1\n    opttype::ToptT = Flux.ADAM\n    opt::Topt = opttype(Flux.params(net))\n    startlearningat::Int64 = 10^3\n    minibatchsize::Int64 = 32\n    doubledqn::Bool = true\n    nmarkov::Int64 = 1\n    nsteps::Int64 = 1\n    replaysize::Int64 = 10^4\n    loss::Function = Flux.mse\n\n\n\n"
},

{
    "location": "learners/#ReinforcementLearning.DeepActorCritic",
    "page": "Learners",
    "title": "ReinforcementLearning.DeepActorCritic",
    "category": "type",
    "text": "mutable struct DeepActorCritic{Tnet, Tpl, Tplm, Tvl, ToptT, Topt}\n    nh::Int64 = 4\n    na::Int64 = 2\n    γ::Float64 = .9\n    nsteps::Int64 = 5\n    net::Tnet\n    policylayer::Tpl = Linear(nh, na)\n    policynet::Tplm = Flux.Chain(Flux.mapleaves(Flux.Tracker.data, net),\n                             Flux.mapleaves(Flux.Tracker.data, policylayer))\n    valuelayer::Tvl = Linear(nh, 1)\n    params::Array{Any, 1} = vcat(map(Flux.params, [net, policylayer, valuelayer])...)\n    t::Int64 = 0\n    updateevery::Int64 = 1\n    opttype::ToptT = Flux.ADAM\n    opt::Topt = opttype(params)\n    αcritic::Float64 = .1\n    nmarkov::Int64 = 1\n\n\n\n"
},

{
    "location": "learners/#Deep-Reinforcement-Learning-1",
    "page": "Learners",
    "title": "Deep Reinforcement Learning",
    "category": "section",
    "text": "Modules = [ReinforcementLearning]\nPages   = [\"dqn.jl\", \"deepactorcritic.jl\"]"
},

{
    "location": "environments/#",
    "page": "Environments",
    "title": "Environments",
    "category": "page",
    "text": ""
},

{
    "location": "environments/#environments-1",
    "page": "Environments",
    "title": "Environments",
    "category": "section",
    "text": "The following environments can be added with (v1.0) pkg> add RLEnvXZYorPkg.add(\"RLEnvXYZ\")Examples can be found in the example folders of these repositories.RLEnvAtari, RLEnvClassicControl, RLEnvDiscrete. RLEnvViZDoom. RLEnvViZGym."
},

{
    "location": "environments/#ReinforcementLearning.MDP",
    "page": "Environments",
    "title": "ReinforcementLearning.MDP",
    "category": "type",
    "text": "mutable struct MDP \n    ns::Int64\n    na::Int64\n    state::Int64\n    trans_probs::Array{AbstractArray, 2}\n    reward::Array{Float64, 2}\n    initialstates::Array{Int64, 1}\n    isterminal::Array{Int64, 1}\n\nA Markov Decision Process with ns states, na actions, current state, naxns - array of transition probabilites trans_props which consists for every (action, state) pair of a (potentially sparse) array that sums to 1 (see getprobvecrandom, getprobvecuniform, getprobvecdeterministic for helpers to constract the transition probabilities) naxns - array of reward, array of initial states initialstates, and ns - array of 0/1 indicating if a state is terminal.\n\n\n\n"
},

{
    "location": "environments/#ReinforcementLearning.MDP-Tuple{Any,Any}",
    "page": "Environments",
    "title": "ReinforcementLearning.MDP",
    "category": "method",
    "text": "MDP(ns, na; init = \"random\")\nMDP(; ns = 10, na = 4, init = \"random\")\n\nReturn MDP with init in (\"random\", \"uniform\", \"deterministic\"), where the keyword init determines how to construct the transition probabilites (see also  getprobvecrandom, getprobvecuniform, getprobvecdeterministic).\n\n\n\n"
},

{
    "location": "environments/#ReinforcementLearning.AbsorbingDetMDP-Tuple{}",
    "page": "Environments",
    "title": "ReinforcementLearning.AbsorbingDetMDP",
    "category": "method",
    "text": "AbsorbingDetMDP(;ns = 10^3, na = 10)\n\nReturns a random deterministic absorbing MDP\n\n\n\n"
},

{
    "location": "environments/#ReinforcementLearning.DetMDP-Tuple{}",
    "page": "Environments",
    "title": "ReinforcementLearning.DetMDP",
    "category": "method",
    "text": "DetMDP(; ns = 10^4, na = 10)\n\nReturns a random deterministic MDP.\n\n\n\n"
},

{
    "location": "environments/#ReinforcementLearning.DetTreeMDP-Tuple{}",
    "page": "Environments",
    "title": "ReinforcementLearning.DetTreeMDP",
    "category": "method",
    "text": "DetTreeMDP(; na = 4, depth = 5)\n\nReturns a treeMDP with random rewards at the leaf nodes.\n\n\n\n"
},

{
    "location": "environments/#ReinforcementLearning.DetTreeMDPwithinrew-Tuple{}",
    "page": "Environments",
    "title": "ReinforcementLearning.DetTreeMDPwithinrew",
    "category": "method",
    "text": "DetTreeMDPwithinrew(; na = 4, depth = 5)\n\nReturns a treeMDP with random rewards.\n\n\n\n"
},

{
    "location": "environments/#ReinforcementLearning.StochMDP-Tuple{}",
    "page": "Environments",
    "title": "ReinforcementLearning.StochMDP",
    "category": "method",
    "text": "StochMDP(; na = 10, ns = 50) = MDP(ns, na)\n\n\n\n"
},

{
    "location": "environments/#ReinforcementLearning.StochTreeMDP-Tuple{}",
    "page": "Environments",
    "title": "ReinforcementLearning.StochTreeMDP",
    "category": "method",
    "text": "StochTreeMDP(; na = 4, depth = 4, bf = 2)\n\nReturns a random stochastic treeMDP with branching factor bf.\n\n\n\n"
},

{
    "location": "environments/#ReinforcementLearning.run!-Tuple{ReinforcementLearning.MDP,Array{Int64,1}}",
    "page": "Environments",
    "title": "ReinforcementLearning.run!",
    "category": "method",
    "text": "run!(mdp::MDP, policy::Array{Int64, 1}) = run!(mdp, policy[mdp.state])\n\n\n\n"
},

{
    "location": "environments/#ReinforcementLearning.run!-Tuple{ReinforcementLearning.MDP,Int64}",
    "page": "Environments",
    "title": "ReinforcementLearning.run!",
    "category": "method",
    "text": "run!(mdp::MDP, action::Int64)\n\nTransition to a new state given action. Returns the new state.\n\n\n\n"
},

{
    "location": "environments/#ReinforcementLearning.setterminalstates!-Tuple{Any,Any}",
    "page": "Environments",
    "title": "ReinforcementLearning.setterminalstates!",
    "category": "method",
    "text": "setterminalstates!(mdp, range)\n\nSets mdp.isterminal[range] .= 1, empties the table of transition probabilities for terminal states and sets the reward for all actions in the terminal state to the same value.\n\n\n\n"
},

{
    "location": "environments/#ReinforcementLearning.treeMDP-Tuple{Any,Any}",
    "page": "Environments",
    "title": "ReinforcementLearning.treeMDP",
    "category": "method",
    "text": "treeMDP(na, depth; init = \"random\", branchingfactor = 3)\n\nReturns a tree structured MDP with na actions and depth of the tree. If init is random, the branchingfactor determines how many possible states a (action, state) pair has. If init = \"deterministic\" the branchingfactor = na.\n\n\n\n"
},

{
    "location": "environments/#ReinforcementLearning.getprobvecdeterministic",
    "page": "Environments",
    "title": "ReinforcementLearning.getprobvecdeterministic",
    "category": "function",
    "text": "getprobvecdeterministic(n, min = 1, max = n)\n\nReturns a SparseVector of length n where one element in min:max has  value 1.\n\n\n\n"
},

{
    "location": "environments/#ReinforcementLearning.getprobvecrandom-Tuple{Any,Any,Any}",
    "page": "Environments",
    "title": "ReinforcementLearning.getprobvecrandom",
    "category": "method",
    "text": "getprobvecrandom(n, min, max)\n\nReturns an array of length n that sums to 1 where all elements outside of min:max are zero.\n\n\n\n"
},

{
    "location": "environments/#ReinforcementLearning.getprobvecrandom-Tuple{Any}",
    "page": "Environments",
    "title": "ReinforcementLearning.getprobvecrandom",
    "category": "method",
    "text": "getprobvecrandom(n)\n\nReturns an array of length n that sums to 1. More precisely, the array is a sample of a Dirichlet distribution with n categories and _1 =   = _n = 1.\n\n\n\n"
},

{
    "location": "environments/#ReinforcementLearning.getprobvecuniform-Tuple{Any}",
    "page": "Environments",
    "title": "ReinforcementLearning.getprobvecuniform",
    "category": "method",
    "text": "getprobvecuniform(n)  = fill(1/n, n)\n\n\n\n"
},

{
    "location": "environments/#MDPs-1",
    "page": "Environments",
    "title": "MDPs",
    "category": "section",
    "text": "Modules = [ReinforcementLearning]\nPages   = [\"mdp.jl\", \"randommdp.jl\"]"
},

{
    "location": "stop/#",
    "page": "Stopping Criteria",
    "title": "Stopping Criteria",
    "category": "page",
    "text": ""
},

{
    "location": "stop/#ReinforcementLearning.ConstantNumberEpisodes",
    "page": "Stopping Criteria",
    "title": "ReinforcementLearning.ConstantNumberEpisodes",
    "category": "type",
    "text": "mutable struct ConstantNumberEpisodes\n    N::Int64\n    counter::Int64\n\nStops learning when the agent has finished \'N\' episodes.\n\n\n\n"
},

{
    "location": "stop/#ReinforcementLearning.ConstantNumberEpisodes-Tuple{Any}",
    "page": "Stopping Criteria",
    "title": "ReinforcementLearning.ConstantNumberEpisodes",
    "category": "method",
    "text": "    ConstantNumbeEpisodes(N) = ConstantNumberEpisodes(N, 0)\n\n\n\n"
},

{
    "location": "stop/#ReinforcementLearning.ConstantNumberSteps",
    "page": "Stopping Criteria",
    "title": "ReinforcementLearning.ConstantNumberSteps",
    "category": "type",
    "text": "mutable struct ConstantNumberSteps\n    T::Int64\n    counter::Int64\n\nStops learning when the agent has taken \'T\' actions.\n\n\n\n"
},

{
    "location": "stop/#ReinforcementLearning.ConstantNumberSteps-Tuple{Any}",
    "page": "Stopping Criteria",
    "title": "ReinforcementLearning.ConstantNumberSteps",
    "category": "method",
    "text": "ConstantNumberSteps(N) = ConstantNumberSteps(N, 0)\n\n\n\n"
},

{
    "location": "stop/#stop-1",
    "page": "Stopping Criteria",
    "title": "Stopping Criteria",
    "category": "section",
    "text": "Modules = [ReinforcementLearning]\nPages   = [\"stoppingcriterion.jl\"]"
},

{
    "location": "preprocessors/#",
    "page": "Preprocessors",
    "title": "Preprocessors",
    "category": "page",
    "text": ""
},

{
    "location": "preprocessors/#ReinforcementLearning.ImageCrop",
    "page": "Preprocessors",
    "title": "ReinforcementLearning.ImageCrop",
    "category": "type",
    "text": "struct ImageCrop\n    xidx::UnitRange{Int64}\n    yidx::UnitRange{Int64}\n\nSelect indices xidx and yidx from a 2 or 3 dimensional array.\n\nExample:\n\nc = ImageCrop(2:5, 3:2:9)\nc([10i + j for i in 1:10, j in 1:10])\n\n\n\n"
},

{
    "location": "preprocessors/#ReinforcementLearning.ImagePreprocessor",
    "page": "Preprocessors",
    "title": "ReinforcementLearning.ImagePreprocessor",
    "category": "type",
    "text": "struct ImagePreprocessor\n    size\n    chain\n\nUse chain to preprocess a grayscale or color image of size = (width, height).\n\nExample:\n\np = ImagePreprocessor((100, 100), \n                      [ImageResizeNearestNeighbour((50, 80)),\n                       ImageCrop(1:30, 10:80),\n                       x -> x ./ 256])\nx = rand(UInt8, 100, 100)\ns = ReinforcementLearning.preprocessstate(p, x)\n\n\n\n"
},

{
    "location": "preprocessors/#ReinforcementLearning.ImageResizeBilinear",
    "page": "Preprocessors",
    "title": "ReinforcementLearning.ImageResizeBilinear",
    "category": "type",
    "text": "struct ImageResizeBilinear\n    outdim::Tuple{Int64, Int64}\n\nResize any image to outdim = (width, height) with bilinear interpolation.\n\nExample:\n\nr = ImageResizeBilinear((50, 50))\nr(rand(200, 200))\nr(rand(UInt8, 3, 100, 100))\n\n\n\n"
},

{
    "location": "preprocessors/#ReinforcementLearning.ImageResizeNearestNeighbour",
    "page": "Preprocessors",
    "title": "ReinforcementLearning.ImageResizeNearestNeighbour",
    "category": "type",
    "text": "struct ImageResizeNearestNeighbour\n    outdim::Tuple{Int64, Int64}\n\nResize any image to outdim = (width, height) by nearest-neighbour interpolation (i.e. subsampling).\n\nExample:\n\nr = ImageResizeNearestNeighbour((50, 50))\nr(rand(200, 200))\nr(rand(UInt8, 3, 100, 100))\n\n\n\n"
},

{
    "location": "preprocessors/#ReinforcementLearning.NoPreprocessor",
    "page": "Preprocessors",
    "title": "ReinforcementLearning.NoPreprocessor",
    "category": "type",
    "text": "struct NoPreprocessor end\n\n\n\n"
},

{
    "location": "preprocessors/#ReinforcementLearning.RadialBasisFunctions",
    "page": "Preprocessors",
    "title": "ReinforcementLearning.RadialBasisFunctions",
    "category": "type",
    "text": "struct RadialBasisFunctions\n    means::Array{Array{Float64, 1}, 1}\n    sigmas::Array{Float64, 1}\n    state::Array{Float64, 1}\n\n\n\n"
},

{
    "location": "preprocessors/#ReinforcementLearning.RandomProjection",
    "page": "Preprocessors",
    "title": "ReinforcementLearning.RandomProjection",
    "category": "type",
    "text": "struct RandomProjection\n    w::Array{Float64, 2}\n\n\n\n"
},

{
    "location": "preprocessors/#ReinforcementLearning.SparseRandomProjection",
    "page": "Preprocessors",
    "title": "ReinforcementLearning.SparseRandomProjection",
    "category": "type",
    "text": "struct SparseRandomProjection\n    w::Array{Float64, 2}\n    b::Array{Float64, 1}\n\n\n\n"
},

{
    "location": "preprocessors/#ReinforcementLearning.StateAggregator",
    "page": "Preprocessors",
    "title": "ReinforcementLearning.StateAggregator",
    "category": "type",
    "text": "struct StateAggregator\n    box::Box\n    ns::Int64\n    nbins::Array{Int64, 1}\n    offsets::Array{Int64, 1}\n    perdimension::Bool\n\n\n\n"
},

{
    "location": "preprocessors/#ReinforcementLearning.StateAggregator-Tuple{Array{T,1} where T,Array{T,1} where T,Array{T,1} where T}",
    "page": "Preprocessors",
    "title": "ReinforcementLearning.StateAggregator",
    "category": "method",
    "text": "StateAggregator(lb::Vector, ub::Vector, nbins::Vector;\n                perdimension = false)\n\n\n\n"
},

{
    "location": "preprocessors/#ReinforcementLearning.StateAggregator-Tuple{Number,Number,Int64,Int64}",
    "page": "Preprocessors",
    "title": "ReinforcementLearning.StateAggregator",
    "category": "method",
    "text": "StateAggregator(lb::Number, ub::Number, nbins::Int, ndims::Int; \n                perdimension = false)\n\n\n\n"
},

{
    "location": "preprocessors/#ReinforcementLearning.TilingStateAggregator",
    "page": "Preprocessors",
    "title": "ReinforcementLearning.TilingStateAggregator",
    "category": "type",
    "text": "struct TilingStateAggregator{T <: Array{StateAggregator,1}}\n    ns::Int64\n    tiling::T\n\n\n\n"
},

{
    "location": "preprocessors/#ReinforcementLearning.Box",
    "page": "Preprocessors",
    "title": "ReinforcementLearning.Box",
    "category": "type",
    "text": "struct Box{T}\n    low::Array{T, 1}\n    high::Array{T, 1}\n\n\n\n"
},

{
    "location": "preprocessors/#preprocessors-1",
    "page": "Preprocessors",
    "title": "Preprocessors",
    "category": "section",
    "text": "Modules = [ReinforcementLearning]\nPages   = [\"preprocessor.jl\"]"
},

{
    "location": "policies/#",
    "page": "Policies",
    "title": "Policies",
    "category": "page",
    "text": ""
},

{
    "location": "policies/#policies-1",
    "page": "Policies",
    "title": "Policies",
    "category": "section",
    "text": ""
},

{
    "location": "policies/#ReinforcementLearning.EpsilonGreedyPolicy",
    "page": "Policies",
    "title": "ReinforcementLearning.EpsilonGreedyPolicy",
    "category": "type",
    "text": "mutable struct EpsilonGreedyPolicy <: AbstractEpsilonGreedyPolicy\n    ϵ::Float64\n\nChooses the action with the highest value with probability 1 - ϵ and selects an action uniformly random with probability ϵ. For states with actions that where never performed before, the behavior of the VeryOptimisticEpsilonGreedyPolicy is followed.\n\n\n\n"
},

{
    "location": "policies/#ReinforcementLearning.OptimisticEpsilonGreedyPolicy",
    "page": "Policies",
    "title": "ReinforcementLearning.OptimisticEpsilonGreedyPolicy",
    "category": "type",
    "text": "mutable struct OptimisticEpsilonGreedyPolicy <: AbstractEpsilonGreedyPolicy\n    ϵ::Float64\n\nEpsilonGreedyPolicy that samples uniformly from the actions with the highest Q-value and novel actions in each state where actions are available that where never chosen before. \n\n\n\n"
},

{
    "location": "policies/#ReinforcementLearning.PesimisticEpsilonGreedyPolicy",
    "page": "Policies",
    "title": "ReinforcementLearning.PesimisticEpsilonGreedyPolicy",
    "category": "type",
    "text": "mutable struct PesimisticEpsilonGreedyPolicy <: AbstractEpsilonGreedyPolicy\n    ϵ::Float64\n\nEpsilonGreedyPolicy that does not handle novel actions differently.\n\n\n\n"
},

{
    "location": "policies/#ReinforcementLearning.VeryOptimisticEpsilonGreedyPolicy",
    "page": "Policies",
    "title": "ReinforcementLearning.VeryOptimisticEpsilonGreedyPolicy",
    "category": "type",
    "text": "mutable struct VeryOptimisticEpsilonGreedyPolicy <: AbstractEpsilonGreedyPolicy\n    ϵ::Float64\n\nEpsilonGreedyPolicy that samples uniformly from novel actions in each state where actions are available that where never chosen before. See also  Initial values, novel actions and unseen values.\n\n\n\n"
},

{
    "location": "policies/#Epsilon-Greedy-Policies-1",
    "page": "Policies",
    "title": "Epsilon Greedy Policies",
    "category": "section",
    "text": "Modules = [ReinforcementLearning]\nPages   = [\"epsilongreedypolicies.jl\"]"
},

{
    "location": "policies/#ReinforcementLearning.AbstractSoftmaxPolicy",
    "page": "Policies",
    "title": "ReinforcementLearning.AbstractSoftmaxPolicy",
    "category": "type",
    "text": "mutable struct SoftmaxPolicy <: AbstractSoftmaxPolicy\n    β::Float64\n\nChoose action a with probability\n\nfrace^beta x_asum_a e^beta x_a\n\nwhere x is a vector of values for each action. In states with actions that were never chosen before, a uniform random novel action is returned.\n\nSoftmaxPolicy(; β = 1.)\n\nReturns a SoftmaxPolicy with default β = 1.\n\n\n\n"
},

{
    "location": "policies/#Softmax-Policies-1",
    "page": "Policies",
    "title": "Softmax Policies",
    "category": "section",
    "text": "Modules = [ReinforcementLearning]\nPages   = [\"softmaxpolicy.jl\"]"
},

{
    "location": "policies/#ReinforcementLearning.ForcedEpisode",
    "page": "Policies",
    "title": "ReinforcementLearning.ForcedEpisode",
    "category": "type",
    "text": "mutable struct ForcedEpisode{Ts}\n    t::Int64\n    states::Ts\n    dones::Array{Bool, 1}\n    rewards::Array{Float64, 1}\n\n\n\n"
},

{
    "location": "policies/#ReinforcementLearning.ForcedPolicy",
    "page": "Policies",
    "title": "ReinforcementLearning.ForcedPolicy",
    "category": "type",
    "text": "mutable struct ForcedPolicy \n    t::Int64\n    actions::Array{Int64, 1}\n\n\n\n"
},

{
    "location": "policies/#Forced-Policy-and-Episode-1",
    "page": "Policies",
    "title": "Forced Policy and Episode",
    "category": "section",
    "text": "Modules = [ReinforcementLearning]\nPages   = [\"forced.jl\"]"
},

{
    "location": "callbacks/#",
    "page": "Callbacks",
    "title": "Callbacks",
    "category": "page",
    "text": ""
},

{
    "location": "callbacks/#ReinforcementLearning.AllRewards",
    "page": "Callbacks",
    "title": "ReinforcementLearning.AllRewards",
    "category": "type",
    "text": "struct AllRewards\n    rewards::Array{Float64, 1}\n\nRecords all rewards.\n\n\n\n"
},

{
    "location": "callbacks/#ReinforcementLearning.AllRewards-Tuple{}",
    "page": "Callbacks",
    "title": "ReinforcementLearning.AllRewards",
    "category": "method",
    "text": "AllRewards()\n\nInitializes with empty array.\n\n\n\n"
},

{
    "location": "callbacks/#ReinforcementLearning.EvaluateGreedy-Tuple{Any,Any}",
    "page": "Callbacks",
    "title": "ReinforcementLearning.EvaluateGreedy",
    "category": "method",
    "text": "EvaluateGreedy(callback, stoppincriterion; every = Episode(10))\n\nEvaluate an rlsetup greedily by leaving the normal learning loop and evaluating the agent with callback until stoppingcriterion is met, at which point normal learning is resumed. This is done every Nth Episode (where N = 10 by default) or every Nth Step (e.g. every = Step(10)).\n\nExample:\n\neg = EvaluateGreedy(EvaluationPerEpisode(TotalReward(), returnmean = true),\n                    ConstantNumberEpisodes(10), every = Episode(100))\nrlsetup = RLSetup(learner, environment, stoppingcriterion, callbacks = [eg])\nlearn!(rlsetup)\ngetvalue(eg)\n\nLeaves the learning loop every 100th episode to estimate the average total reward per episode, by running a greedy policy for 10 episodes.\n\n\n\n"
},

{
    "location": "callbacks/#ReinforcementLearning.LinearDecreaseEpsilon",
    "page": "Callbacks",
    "title": "ReinforcementLearning.LinearDecreaseEpsilon",
    "category": "type",
    "text": "mutable struct LinearDecreaseEpsilon\n    start::Int64\n    stop::Int64\n    initval::Float64\n    finalval::Float64\n    t::Int64\n    step::Float64\n\nLinearly decrease ϵ of an EpsilonGreedyPolicy from initval until  step start to finalval at step stop.\n\nStepsize step = (finalval - initval)/(stop - start).\n\n\n\n"
},

{
    "location": "callbacks/#ReinforcementLearning.LinearDecreaseEpsilon-NTuple{4,Any}",
    "page": "Callbacks",
    "title": "ReinforcementLearning.LinearDecreaseEpsilon",
    "category": "method",
    "text": "LinearDecreaseEpsilon(start, stop, initval, finalval)\n\n\n\n"
},

{
    "location": "callbacks/#ReinforcementLearning.Progress",
    "page": "Callbacks",
    "title": "ReinforcementLearning.Progress",
    "category": "type",
    "text": "mutable struct Progress \n    steps::Int64\n    laststopcountervalue::Int64\n\nShow steps times progress information during learning.\n\n\n\n"
},

{
    "location": "callbacks/#ReinforcementLearning.Progress",
    "page": "Callbacks",
    "title": "ReinforcementLearning.Progress",
    "category": "type",
    "text": "Progress(steps = 10) = Progress(steps, 0)\n\n\n\n"
},

{
    "location": "callbacks/#ReinforcementLearning.RecordAll",
    "page": "Callbacks",
    "title": "ReinforcementLearning.RecordAll",
    "category": "type",
    "text": "struct RecordAll\n    rewards::Array{Float64, 1}\n    actions::Array{Int64, 1}\n    states::Array{Int64, 1}\n    done::Array{Bool, 1}\n\nRecords everything.\n\n\n\n"
},

{
    "location": "callbacks/#ReinforcementLearning.RecordAll-Tuple{}",
    "page": "Callbacks",
    "title": "ReinforcementLearning.RecordAll",
    "category": "method",
    "text": "RecordAll()\n\nInitializes with empty arrays.\n\n\n\n"
},

{
    "location": "callbacks/#ReinforcementLearning.ReduceEpsilonPerEpisode",
    "page": "Callbacks",
    "title": "ReinforcementLearning.ReduceEpsilonPerEpisode",
    "category": "type",
    "text": "mutable struct ReduceEpsilonPerEpisode\n    ϵ0::Float64\n    counter::Int64\n\nReduces ϵ of an EpsilonGreedyPolicy after each episode.\n\nIn episode n, ϵ = ϵ0/n\n\n\n\n"
},

{
    "location": "callbacks/#ReinforcementLearning.ReduceEpsilonPerEpisode-Tuple{}",
    "page": "Callbacks",
    "title": "ReinforcementLearning.ReduceEpsilonPerEpisode",
    "category": "method",
    "text": "ReduceEpsilonPerEpisode()\n\nInitialize callback.\n\n\n\n"
},

{
    "location": "callbacks/#ReinforcementLearning.ReduceEpsilonPerT",
    "page": "Callbacks",
    "title": "ReinforcementLearning.ReduceEpsilonPerT",
    "category": "type",
    "text": "mutable struct ReduceEpsilonPerT\n    ϵ0::Float64\n    T::Int64\n    n::Int64\n    counter::Int64\n\nReduces ϵ of an EpsilonGreedyPolicy after every T steps.\n\nAfter n * T steps, ϵ = ϵ0/n\n\n\n\n"
},

{
    "location": "callbacks/#ReinforcementLearning.ReduceEpsilonPerT-Tuple{Any}",
    "page": "Callbacks",
    "title": "ReinforcementLearning.ReduceEpsilonPerT",
    "category": "method",
    "text": "ReduceEpsilonPerT()\n\nInitialize callback.\n\n\n\n"
},

{
    "location": "callbacks/#ReinforcementLearning.SaveLearner",
    "page": "Callbacks",
    "title": "ReinforcementLearning.SaveLearner",
    "category": "type",
    "text": "@with_kw struct SaveLearner{T}\n    every::T = Step(10^3)\n    filename::String = tempname()\n\nSave learner every Nth Step (or Nth Episode) to filename_i.jld2, where i is the step (or episode) at which the learner is saved.\n\n\n\n"
},

{
    "location": "callbacks/#ReinforcementLearning.Visualize",
    "page": "Callbacks",
    "title": "ReinforcementLearning.Visualize",
    "category": "type",
    "text": "mutable struct Visualize \n    plot\n    wait::Float64\n\n\n\n"
},

{
    "location": "callbacks/#ReinforcementLearning.Visualize-Tuple{}",
    "page": "Callbacks",
    "title": "ReinforcementLearning.Visualize",
    "category": "method",
    "text": "Visualize(; wait = .15)\n\nA callback to be used in an RLSetup to visualize an environment during  running or learning.\n\n\n\n"
},

{
    "location": "callbacks/#callbacks-1",
    "page": "Callbacks",
    "title": "Callbacks",
    "category": "section",
    "text": "Modules = [ReinforcementLearning]\nPages   = [\"callbacks.jl\"]"
},

{
    "location": "metrics/#",
    "page": "Evaluation Metrics",
    "title": "Evaluation Metrics",
    "category": "page",
    "text": ""
},

{
    "location": "metrics/#ReinforcementLearning.EvaluationPerEpisode",
    "page": "Evaluation Metrics",
    "title": "ReinforcementLearning.EvaluationPerEpisode",
    "category": "type",
    "text": "EvaluationPerEpisode\n    values::Array{Float64, 1}\n    metric::SimpleEvaluationMetric\n\nStores the value of the simple metric for each episode in values.\n\n\n\n"
},

{
    "location": "metrics/#ReinforcementLearning.EvaluationPerEpisode",
    "page": "Evaluation Metrics",
    "title": "ReinforcementLearning.EvaluationPerEpisode",
    "category": "type",
    "text": "EvaluationPerEpisode(metric = MeanReward())\n\nInitializes with empty values array and simple metric (default MeanReward). Other options are TimeSteps (to measure the lengths of episodes) or TotalReward.\n\n\n\n"
},

{
    "location": "metrics/#ReinforcementLearning.EvaluationPerT",
    "page": "Evaluation Metrics",
    "title": "ReinforcementLearning.EvaluationPerT",
    "category": "type",
    "text": "EvaluationPerT\n    T::Int64\n    counter::Int64\n    values::Array{Float64, 1}\n    metric::SimpleEvaluationMetric\n\nStores the value of the simple metric after every T steps in values.\n\n\n\n"
},

{
    "location": "metrics/#ReinforcementLearning.EvaluationPerT",
    "page": "Evaluation Metrics",
    "title": "ReinforcementLearning.EvaluationPerT",
    "category": "type",
    "text": "EvaluationPerT(T, metric = MeanReward())\n\nInitializes with T, counter = 0, empty values array and simple metric (default MeanReward).  Another option is TotalReward.\n\n\n\n"
},

{
    "location": "metrics/#ReinforcementLearning.MeanReward",
    "page": "Evaluation Metrics",
    "title": "ReinforcementLearning.MeanReward",
    "category": "type",
    "text": "mutable struct MeanReward \n    meanreward::Float64\n    counter::Int64\n\nComputes iteratively the mean reward.\n\n\n\n"
},

{
    "location": "metrics/#ReinforcementLearning.MeanReward-Tuple{}",
    "page": "Evaluation Metrics",
    "title": "ReinforcementLearning.MeanReward",
    "category": "method",
    "text": "MeanReward()\n\nInitializes counter and meanreward to 0.\n\n\n\n"
},

{
    "location": "metrics/#ReinforcementLearning.TimeSteps",
    "page": "Evaluation Metrics",
    "title": "ReinforcementLearning.TimeSteps",
    "category": "type",
    "text": "mutable struct TimeSteps\n    counter::Int64\n\nCounts the number of timesteps the simulation is running.\n\n\n\n"
},

{
    "location": "metrics/#ReinforcementLearning.TimeSteps-Tuple{}",
    "page": "Evaluation Metrics",
    "title": "ReinforcementLearning.TimeSteps",
    "category": "method",
    "text": "TimeSteps()\n\nInitializes counter to 0.\n\n\n\n"
},

{
    "location": "metrics/#ReinforcementLearning.TotalReward",
    "page": "Evaluation Metrics",
    "title": "ReinforcementLearning.TotalReward",
    "category": "type",
    "text": "mutable struct TotalReward \n    reward::Float64\n\nAccumulates all rewards.\n\n\n\n"
},

{
    "location": "metrics/#ReinforcementLearning.TotalReward-Tuple{}",
    "page": "Evaluation Metrics",
    "title": "ReinforcementLearning.TotalReward",
    "category": "method",
    "text": "TotalReward()\n\nInitializes reward to 0.\n\n\n\n"
},

{
    "location": "metrics/#metrics-1",
    "page": "Evaluation Metrics",
    "title": "Evaluation Metrics",
    "category": "section",
    "text": "Modules = [ReinforcementLearning]\nPages   = [\"metrics.jl\"]"
},

{
    "location": "api/#",
    "page": "API",
    "title": "API",
    "category": "page",
    "text": ""
},

{
    "location": "api/#API-1",
    "page": "API",
    "title": "API",
    "category": "section",
    "text": "New learners, policies, callbacks, environments, evaluation metrics or stopping criteria need to implement the following functions."
},

{
    "location": "api/#Learners-1",
    "page": "API",
    "title": "Learners",
    "category": "section",
    "text": "update!(learner, buffer)Returns nothing.selectaction(learner, policy, state)Returns an action.defaultbuffer(learner, environment, preprocessor)Returns nothing."
},

{
    "location": "api/#Policies-1",
    "page": "API",
    "title": "Policies",
    "category": "section",
    "text": "selectaction(policy, values)Returns an action.getactionprobabilities(policy, state)Returns a normalized (1-norm) vector with non-negative entries."
},

{
    "location": "api/#Callbacks-1",
    "page": "API",
    "title": "Callbacks",
    "category": "section",
    "text": "callback!(callback, rlsetup, state, action, reward, done)Returns nothing."
},

{
    "location": "api/#api_environments-1",
    "page": "API",
    "title": "Environments",
    "category": "section",
    "text": "interact!(action, environment)Returns state, reward, done.getstate(environment)Returns state, done.reset!(environment)Returns nothing."
},

{
    "location": "api/#getvalue-1",
    "page": "API",
    "title": "Evaluation Metrics",
    "category": "section",
    "text": "getvalue(metric)Any return value allowed."
},

{
    "location": "api/#Stopping-Criteria-1",
    "page": "API",
    "title": "Stopping Criteria",
    "category": "section",
    "text": "isbreak!(stoppingcriterion, state, action, reward, done)Returns true or false."
},

]}
