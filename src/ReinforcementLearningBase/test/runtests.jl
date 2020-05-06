using ReinforcementLearningBase
using Test

include("lottery_env.jl")

@testset "ReinforcementLearningBase" begin
    include("spaces.jl")
    include("base.jl")
end
