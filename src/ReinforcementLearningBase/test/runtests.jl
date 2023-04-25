using ReinforcementLearningBase
using Test

using CommonRLInterface
const CRL = CommonRLInterface

using POMDPs
using POMDPTools: Deterministic

@testset "ReinforcementLearningBase" begin
    include("CommonRLInterface.jl")
    include("interface.jl")
end
