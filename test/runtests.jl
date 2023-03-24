using Test
using ReinforcementLearning

const SRC = (@__DIR__)* "/../src/"
@testset "ReinforcementLearning" begin
    include(SRC*"/ReinforcementLearningCore/test/runtests.jl")
    include(SRC*"/ReinforcementLearningBase/test/runtests.jl")
    include(SRC*"/ReinforcementLearningZoo/test/runtests.jl")
#    include(SRC*"/ReinforcementLearningEnvironments/test/runtests.jl")
#    include(SRC*"/DistributedReinforcementLearning/test/runtests.jl")
end
