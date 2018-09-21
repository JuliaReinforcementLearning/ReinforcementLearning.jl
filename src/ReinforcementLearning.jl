module ReinforcementLearning

using DataStructures, Parameters, SparseArrays, LinearAlgebra, Distributed,
Statistics, Dates, Requires, StatsBase
import Statistics: mean
import ReinforcementLearningBase: interact!, getstate, reset!, plotenv,
actionspace, sample


using Random: seed!
function __init__()
    @require CuArrays="3a865a2d-5b23-5a0f-bc46-62713ec82fae" begin
    """
        togpu(x)

    Send array `x` to GPU. Requires the `using CuArrays`.
    """
    togpu(x) = CuArrays.adapt(CuArrays.CuArray, x)
    end
end


include("helper.jl")
include("buffers.jl")
include("traces.jl")
include("policies.jl")
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
