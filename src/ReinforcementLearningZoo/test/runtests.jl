using ReinforcementLearningZoo
using Test
using ReinforcementLearningBase
using ReinforcementLearningCore
using Flux
using StatsBase

@testset "ReinforcementLearningZoo.jl" begin

    include("basic_dqn.jl")
    include("dqn.jl")
    include("prioritized_dqn.jl")
    include("rainbow.jl")

end
