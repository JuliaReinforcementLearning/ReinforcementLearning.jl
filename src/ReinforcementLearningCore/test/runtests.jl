using ReinforcementLearningBase
using ReinforcementLearningCore
using ReinforcementLearningEnvironments
using ReinforcementLearningTrajectories

using Test
using CUDA
using CircularArrayBuffers
using Flux

@testset "ReinforcementLearningCore.jl" begin
    include("core/core.jl")
    include("core/stop_conditions.jl")
    include("policies/policies.jl")
    include("utils/utils.jl")
end
