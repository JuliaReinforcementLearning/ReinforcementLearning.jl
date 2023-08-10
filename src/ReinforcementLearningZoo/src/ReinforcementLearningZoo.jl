module ReinforcementLearningZoo

using ReinforcementLearningBase
using ReinforcementLearningCore

const RLZoo = ReinforcementLearningZoo
export RLZoo

include("algorithms/algorithms.jl")
# include("hooks/hooks.jl") # TotalBatchRewardPerEpisode is broken, need to ensure vector copy works!

end # module
