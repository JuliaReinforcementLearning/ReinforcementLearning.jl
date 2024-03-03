using Test
using UUIDs
using Preferences

if Sys.isapple() && Sys.ARCH === :aarch64
    flux_uuid = UUID("587475ba-b771-5e3f-ad9e-33799f191a9c")
    set_preferences!(flux_uuid, "gpu_backend" => "Metal")

    using Metal
else
    using CUDA, cuDNN
end

using ReinforcementLearningBase
using ReinforcementLearningCore
using ReinforcementLearningTrajectories

using CircularArrayBuffers
using Flux

@info "Flux.GPU_BACKEND = $(Flux.GPU_BACKEND)"

include("environments/randomwalk1D.jl")
include("environments/tictactoe.jl")
include("environments/rockpaperscissors.jl")
@testset "ReinforcementLearningCore.jl" begin
    include("core/core.jl")
    include("core/stop_conditions.jl")
    include("policies/policies.jl")
    include("utils/utils.jl")
end
