module ReinforcementLearningZoo

using ReinforcementLearningBase
using ReinforcementLearningCore
import ReinforcementLearningCore.forward
const RLZoo = ReinforcementLearningZoo
export RLZoo
import MLUtils

@warn "ReinforcementLearningZoo is deprecated! Components compatible with ReinforcementLearning v0.11+ are available in ReinforcementLearningFarm."

include("algorithms/algorithms.jl")
# include("hooks/hooks.jl") # TotalBatchRewardPerEpisode is broken, need to ensure vector copy works!

end # module
