module ReinforcementLearningZoo

using ReinforcementLearningBase
using ReinforcementLearningCore
using KernelAbstractions

const RLZoo = ReinforcementLearningZoo
export RLZoo

include("algorithms/algorithms.jl")

end # module
