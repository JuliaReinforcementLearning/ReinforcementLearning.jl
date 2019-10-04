include("StatsBase.jl")
include("ReinforcementLearningEnvironments.jl")
include("Flux.jl")

using Flux

if Flux.has_cuarrays()
    include("CuArrays.jl")
end