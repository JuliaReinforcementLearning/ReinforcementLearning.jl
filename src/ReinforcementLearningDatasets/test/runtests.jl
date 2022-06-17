using ReinforcementLearningDatasets
using DataDeps
using Test
using Random

ENV["DATADEPS_ALWAYS_ACCEPT"] = "true"

@testset "ReinforcementLearningDatasets.jl" begin
    include("dataset.jl")
    include("d4rl_pybullet.jl")
    include("rl_unplugged_atari.jl")
    include("bsuite.jl")
    include("rl_unplugged_dm.jl")
    # include("deep_ope_d4rl.jl")
    # include("atari_dataset.jl")
end
