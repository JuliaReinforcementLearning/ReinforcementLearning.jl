module ReinforcementLearningFarm

using ReinforcementLearning
const RLFarm = ReinforcementLearningFarm
export RLFarm

include("algorithms/algorithms.jl")
include("hooks/hooks.jl")

end # module
