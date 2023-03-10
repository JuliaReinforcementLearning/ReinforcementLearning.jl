module MultiAgentReinforcementLearning

export MultiAgentRL
const MultiAgentRL = MultiAgentReinforcementLearning


using ReinforcementLearningBase
using ReinforcementLearningCore
using Random
using Flux
using Statistics

include("policies/multi_agent.jl")


end 
