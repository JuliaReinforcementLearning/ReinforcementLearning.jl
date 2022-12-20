module ReinforcementLearning

export RL
const RL = ReinforcementLearning

using Reexport

include("devmode.jl")

@reexport using ReinforcementLearningBase
@reexport using ReinforcementLearningCore
@reexport using ReinforcementLearningEnvironments
@reexport using ReinforcementLearningZoo

end
