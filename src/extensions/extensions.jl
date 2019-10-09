include("StatsBase.jl")
include("ReinforcementLearningEnvironments.jl")
include("Flux.jl")
include("Zygote.jl")
include("Knet.jl")

using CUDAapi

if has_cuda()
    include("CuArrays.jl")
end