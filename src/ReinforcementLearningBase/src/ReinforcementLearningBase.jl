module ReinforcementLearningBase

const RLBase = ReinforcementLearningBase
export RLBase

using Random
using Reexport

include("inline_export.jl")
include("interface.jl")
include("CommonRLInterface.jl")
include("base.jl")
include("space.jl")

end # module
