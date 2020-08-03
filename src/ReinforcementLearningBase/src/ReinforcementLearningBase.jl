module ReinforcementLearningBase

const RLBase = ReinforcementLearningBase
export RLBase

include("inline_export.jl")
include("interface.jl")
include("implementations/implementations.jl")
include("base.jl")
include("CommonRLInterface.jl")

end # module
