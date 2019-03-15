using Test
using ReinforcementLearningEnvironments
using POMDPModels

include("utils.jl")

@testset "ReinforcementLearningEnvironments" begin

include("spaces.jl")
include("environments/environments.jl")

end