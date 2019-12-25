using ReinforcementLearningCore
using Random
using Test
using StatsBase

@testset "ReinforcementLearningCore.jl" begin
    include("core/core.jl")
    include("components/components.jl")
    include("utils/utils.jl")
end
