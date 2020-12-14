using ReinforcementLearningBase
using Random
using StableRNGs
using Test
using Statistics
import ReinforcementLearningBase: CRL
import ReinforcementLearningBase.CRL

@testset "ReinforcementLearningBase" begin
    include("examples/examples.jl")
    include("common_rl_env.jl")
end
