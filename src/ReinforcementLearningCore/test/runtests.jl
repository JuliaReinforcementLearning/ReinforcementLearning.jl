using ReinforcementLearningBase
using ReinforcementLearningCore
using ReinforcementLearningEnvironments
using ReinforcementLearningTrajectories

using Test
using UUIDs
using Preferences


if Sys.isapple()
    flux_uuid = UUID("587475ba-b771-5e3f-ad9e-33799f191a9c")
    set_preferences!(flux_uuid, "gpu_backend" => "Metal")

    using Metal
else
    using CUDA, cuDNN
end

using CircularArrayBuffers
using Flux
println("Flux.GPU_BACKEND = $(Flux.GPU_BACKEND)")

@testset "ReinforcementLearningCore.jl" begin
    include("core/core.jl")
    include("core/stop_conditions.jl")
    include("policies/policies.jl")
    include("utils/utils.jl")
end
