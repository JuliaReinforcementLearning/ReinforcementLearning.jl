module ReinforcementLearningCore

using Reexport

const RLCore = ReinforcementLearningCore
export RLCore

@reexport using ReinforcementLearningBase

include("utils/utils.jl")
include("core/core.jl")
include("components/components.jl")

end # module
