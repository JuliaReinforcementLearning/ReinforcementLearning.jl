module ReinforcementLearningBase

const RLBase = ReinforcementLearningBase
export RLBase

using Random
using Reexport

@reexport using CommonRLSpaces

include("inline_export.jl")
include("interface.jl")
include("CommonRLInterface.jl")
include("base.jl")

end # module
