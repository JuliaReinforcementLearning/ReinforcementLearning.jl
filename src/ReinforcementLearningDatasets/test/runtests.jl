using ReinforcementLearningDatasets
using DataDeps
using StableRNGs
using Test

ENV["DATADEPS_ALWAYS_ACCEPT"] = "true"

@testset "ReinforcementLearningDatasets.jl" begin
    include("d4rl/d4rl_dataset.jl")
end
