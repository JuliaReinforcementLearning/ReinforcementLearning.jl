module ReinforcementLearningZoo

using ReinforcementLearningBase
using ReinforcementLearningCore
import ReinforcementLearningCore.forward
const RLZoo = ReinforcementLearningZoo
export RLZoo
import MLUtils

include("utils/device.jl")
include("algorithms/algorithms.jl")
# include("hooks/hooks.jl") # TotalBatchRewardPerEpisode is broken, need to ensure vector copy works!

end # module
