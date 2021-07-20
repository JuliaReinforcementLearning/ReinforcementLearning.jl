using Base: NamedTuple_typename, NamedTuple
using ReinforcementLearningDatasets
using ReinforcementLearningCore: SART, SARTS
using Test

@testset "ReinforcementLearningDatasets.jl" begin
    include("3rd_party/3rd_party_datasets.jl")
end
