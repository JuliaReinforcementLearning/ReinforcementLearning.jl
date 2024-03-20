module ReinforcementLearningFarm

using ReinforcementLearningBase
using ReinforcementLearningCore
const RLFarm = ReinforcementLearningFarm
export RLFarm

include("algorithms/algorithms.jl")
include("hooks/hooks.jl")

end # module
