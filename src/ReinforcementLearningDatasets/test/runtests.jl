using ReinforcementLearningDatasets
using DataDeps
using StableRNGs
using Test

ENV["DATADEPS_ALWAYS_ACCEPT"] = "true"

@testset "ReinforcementLearningDatasets.jl" begin
    include("dataset.jl")
    include("d4rl_pybullet.jl")
    # include("atari_dataset.jl")
end
