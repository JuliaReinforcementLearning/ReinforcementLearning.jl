module ReinforcementLearning

export RL
const RL = ReinforcementLearning

using Reexport

@reexport using ReinforcementLearningBase
@reexport using ReinforcementLearningCore
@reexport using ReinforcementLearningEnvironments

end
