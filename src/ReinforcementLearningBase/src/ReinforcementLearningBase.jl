module ReinforcementLearningBase

export RLBase

"Abbreviated namespace for `ReinforcementLearningBase`."
const RLBase = ReinforcementLearningBase

using Random
using Reexport

include("inline_export.jl")
include("interface.jl")
include("CommonRLInterface.jl")
include("base.jl")
include("space.jl")

end # module
