using ReinforcementLearningBase
using ReinforcementLearningCore
using ReinforcementLearningEnvironments
using ReinforcementLearningTrajectories

using Test
using CUDA
using CircularArrayBuffers
using Flux

@testset "ReinforcementLearningCore.jl" begin
    include("core.jl")
    include("utils/utils.jl")
end
