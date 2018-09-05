VERSION < v"0.7.0-beta2.199" && __precompile__()
module ReinforcementLearning

using DataStructures, Parameters, Compat.SparseArrays, Compat.LinearAlgebra,
Compat.Distributed, Compat.Statistics, Compat.Dates, Compat, Requires
import StatsBase
using Compat: rmul!, @info
using Compat.Statistics: mean

if VERSION < v"0.7.0-beta2.199" 
    # these are ugly hacks for compatibility
    macro distributed(x...); :(@parallel($(esc(x[1])), $(esc(x[2])))) end
    import Compat.foldl
    foldl(op::Function, itr; init = 0) = foldl(op, init, itr) 
    const seed! = srand
    @require CuArrays begin
    """
        togpu(x)

    Send array `x` to GPU. Requires the `using CuArrays`.
    """
    togpu(x) = CuArrays.adapt(CuArrays.CuArray, x)
    end
else
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
end


include("helper.jl")
include("buffers.jl")
include("traces.jl")
include("policies.jl")
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
