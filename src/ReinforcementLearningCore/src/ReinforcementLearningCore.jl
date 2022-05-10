module ReinforcementLearningCore

using ReinforcementLearningBase

const RLCore = ReinforcementLearningCore

export RLCore

include("core/core.jl")
include("policies/policies.jl")
include("utils/utils.jl")

end # module
