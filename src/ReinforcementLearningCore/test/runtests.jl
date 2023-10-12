using ReinforcementLearningBase
using ReinforcementLearningCore
using ReinforcementLearningTrajectories

using Test
using CUDA
using CircularArrayBuffers
using Flux

include("environments/randomwalk1D.jl")
include("environments/tictactoe.jl")
include("environments/rockpaperscissors.jl")
@testset "ReinforcementLearningCore.jl" begin
    include("core/core.jl")
    include("core/stop_conditions.jl")
    include("policies/policies.jl")
    include("utils/utils.jl")
end
