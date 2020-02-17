module ReinforcementLearningCore

using ReinforcementLearningBase

const RLCore = ReinforcementLearningCore
export RLCore

include("utils/utils.jl")
include("core/core.jl")
include("components/components.jl")

end # module
