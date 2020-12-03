module ReinforcementLearningCore

using ReinforcementLearningBase

const RLCore = ReinforcementLearningCore

@doc """
[ReinforcementLearningCore.jl](https://github.com/JuliaReinforcementLearning/ReinforcementLearningCore.jl) (**RLCore**)
provides some standard and reusable components defined by [**RLBase**](https://github.com/JuliaReinforcementLearning/ReinforcementLearningBase.jl), hoping that they are useful for people to implement and experiment with different kinds of algorithms.
""" RLCore

export RLCore

include("utils/utils.jl")
include("extensions/extensions.jl")
include("policies/policies.jl")
include("core/core.jl")

end # module
