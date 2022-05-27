using ReinforcementLearningBase
using ReinforcementLearningCore
using ReinforcementLearningEnvironments
using Trajectories

using Test

@testset "ReinforcementLearningCore.jl" begin
    include("core.jl")
    include("utils/utils.jl")
end
