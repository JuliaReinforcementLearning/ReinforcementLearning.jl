module ReinforcementLearningDatasets

const RLDatasets = ReinforcementLearningDatasets
export RLDatasets

using DataDeps

include("d4rl/register.jl")
include("d4rl_pybullet/register.jl")
include("init.jl")
include("dataset.jl")

end