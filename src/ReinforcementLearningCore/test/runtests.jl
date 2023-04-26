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
    include("core/hooks.jl")
    include("core/stop_conditions.jl")
    include("policies/agent.jl")
    include("policies/multi_agent.jl")
    include("policies/q_based_policy.jl")
    include("utils/utils.jl")
end
