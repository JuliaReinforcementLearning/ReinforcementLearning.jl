module MultiAgentReinforcementLearning

export MultiAgentRL
const MultiAgentRL = MultiAgentReinforcementLearning


using ReinforcementLearningBase
using ReinforcementLearningCore
using ReinforcementLearningTrajectories
using Random
using Flux
using Statistics

include("policies/multi_agent.jl")
include("policies/maddpg.jl")
include("policies/DIAL.jl")

#include("extensions/MessageTrajectory.jl")
include("extensions/CoordGraph.jl")

end
