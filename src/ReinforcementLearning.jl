module ReinforcementLearning

export RL
const RL = ReinforcementLearning

using Reexport

include("ReinforcementLearningBase/src/ReinforcementLearningBase.jl")
include("ReinforcementLearningCore/src/ReinforcementLearningCore.jl")
include("ReinforcementLearningEnvironments/src/ReinforcementLearningEnvironments.jl")
include("ReinforcementLearningZoo/src/ReinforcementLearningZoo.jl")

@reexport using .ReinforcementLearningBase
@reexport using .ReinforcementLearningCore
@reexport using .ReinforcementLearningEnvironments
@reexport using .ReinforcementLearningZoo

end
