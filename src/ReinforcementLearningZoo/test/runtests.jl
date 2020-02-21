using ReinforcementLearningZoo
using Test
using ReinforcementLearningBase
using ReinforcementLearningCore
using Flux
using CUDAapi
using StatsBase

if has_cuda()
    using CuArrays
end

@testset "ReinforcementLearningZoo.jl" begin

include("basic_dqn.jl")

end
