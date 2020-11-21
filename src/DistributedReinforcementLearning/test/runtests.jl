using DistributedReinforcementLearning
using Test
using ReinforcementLearningBase
using ReinforcementLearningCore
using ReinforcementLearningZoo
using ReinforcementLearningEnvironments
using Flux

@testset "DistributedReinforcementLearning.jl" begin

include("actor.jl")
include("core.jl")
# include("example.jl")

end
