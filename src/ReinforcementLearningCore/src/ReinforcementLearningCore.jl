module ReinforcementLearningCore

using ReinforcementLearningBase

const RLCore = ReinforcementLearningCore

@doc """
[ReinforcementLearningCore.jl](https://juliareinforcementlearning.org/docs/rlcore/) (**RLCore**)
provides some standard and reusable components defined by [**RLBase**](https://juliareinforcementlearning.org/docs/rlbase/), hoping that they are useful for people to implement and experiment with different kinds of algorithms.
""" RLCore

export RLCore

include("extensions/extensions.jl")
include("core/core.jl")
include("policies/policies.jl")
include("utils/utils.jl")

end # module
