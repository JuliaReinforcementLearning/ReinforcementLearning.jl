using Test
using ReinforcementLearningEnvironments

include("utils.jl")

@testset "ReinforcementLearningEnvironments" begin

include("spaces.jl")
include("environments/environments.jl")

end