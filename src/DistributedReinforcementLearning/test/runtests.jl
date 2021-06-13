using DistributedReinforcementLearning
using Test
using Distributed
using ReinforcementLearningBase
using ReinforcementLearningCore
using ReinforcementLearningZoo
using ReinforcementLearningEnvironments
using Flux

@testset "DistributedReinforcementLearning.jl" begin

include("actor.jl")
include("core.jl")

end
