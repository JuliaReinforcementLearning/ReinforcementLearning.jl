__precompile__()

module ReinforcementLearning

using DataStructures, Parameters
include("helper.jl")
include("buffers.jl")
include("traces.jl")
include("epsilongreedypolicies.jl")
include("softmaxpolicy.jl")
include("mdp.jl")
include("metrics.jl")
include("stoppingcriterion.jl")
include("callbacks.jl")
include("preprocessor.jl")
include("flux.jl")
for (root, dirs, files) in walkdir(joinpath(@__DIR__, "learner"))
    for file in files
        if splitext(file)[end] == ".jl"
#             println("including $(joinpath(root, file)).")
            include(joinpath(root, file))
        end
    end
end
include("forced.jl")
include("rlsetup.jl")
include("learn.jl")

include("environments.jl")
include("compare.jl")
    

end # module
