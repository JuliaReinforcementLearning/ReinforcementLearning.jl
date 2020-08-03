using ReinforcementLearningBase
using Test
import ReinforcementLearningBase:CRL
import ReinforcementLearningBase.CRL

include("lottery_env.jl")

@testset "ReinforcementLearningBase" begin
    include("spaces.jl")
    include("base.jl")
    include("common_rl_env.jl")
end
