module ReinforcementLearning

using DataStructures, Parameters

include("helper.jl")
include("buffers.jl")
include("traces.jl")
include("epsilongreedypolicies.jl")
include("softmaxpolicy.jl")
include(joinpath("mdp", "mdp.jl"))
include(joinpath("mdp", "randommdp.jl"))
include("metrics.jl")
include("stoppingcriterion.jl")
include("callbacks.jl")
include("preprocessor.jl")
include("flux.jl")
include(joinpath("learner", "montecarlo.jl"))
include(joinpath("learner", "mdplearner.jl"))
include(joinpath("learner", "policygradientlearning.jl"))
include(joinpath("learner", "tdlearning.jl"))
include(joinpath("learner", "prioritizedsweeping.jl"))
include(joinpath("learner", "deepactorcritic.jl"))
include(joinpath("learner", "dqn.jl"))
include("forced.jl")
include("rlsetup.jl")
include("learn.jl")
include("compare.jl")
    

end # module
