using Test
using ReinforcementLearning
using ReinforcementLearning.Utils

using StatsBase
using Random
using Flux

@testset "ReinforcementLearning" begin
    include("Utils/utils.jl")
    include("components/components.jl")
    include("glue/glue.jl")
end