module ReinforcementLearningDatasets

const RLDatasets = ReinforcementLearningDatasets
export RLDatasets

using DataDeps

include("d4rl/register.jl")
include("d4rl/d4rl_dataset.jl")

end