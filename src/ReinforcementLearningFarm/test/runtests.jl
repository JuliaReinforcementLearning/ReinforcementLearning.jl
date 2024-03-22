using UUIDs
using Preferences

if Sys.isapple()
    flux_uuid = UUID("587475ba-b771-5e3f-ad9e-33799f191a9c")
    set_preferences!(flux_uuid, "gpu_backend" => "Metal")

    using Metal
else
    using CUDA, cuDNN
    CUDA.allowscalar(false)
end

using Test
using ReinforcementLearning
using ReinforcementLearningFarm

@testset "ReinforcementLearningFarm.jl" begin
    include("algorithms/algorithms.jl")
    include("hooks/hooks.jl")
end
