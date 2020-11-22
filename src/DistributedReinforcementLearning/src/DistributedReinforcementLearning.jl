module DistributedReinforcementLearning

using Distributed
using ReinforcementLearningBase
using ReinforcementLearningCore

include("actor_model.jl")
include("core.jl")
include("extensions.jl")

end
