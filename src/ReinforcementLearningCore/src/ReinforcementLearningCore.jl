module ReinforcementLearningCore

using TimerOutputs
using ReinforcementLearningBase
using Reexport
const RLCore = ReinforcementLearningCore

export RLCore

@reexport using ReinforcementLearningTrajectories

include("show.jl")
include("core/core.jl")
include("policies/policies.jl")
include("utils/utils.jl")

# Global timer for TimerOutputs.jl
const timer = TimerOutput()

end # module
