module ReinforcementLearningZoo

const RLZoo = ReinforcementLearningZoo
export RLZoo

using ReinforcementLearningBase
using ReinforcementLearningCore
using Setfield: @set
using StableRNGs
using Logging
using Flux.Losses
using Dates

include("patch.jl")
include("algorithms/algorithms.jl")

using Requires

# dynamic loading environments
function __init__()
    @require ReinforcementLearningEnvironments = "25e41dd2-4622-11e9-1641-f1adca772921" begin
        include("experiments/rl_envs/rl_envs.jl")
        @require ArcadeLearningEnvironment = "b7f77d8d-088d-5e02-8ac0-89aab2acc977" include("experiments/atari/atari.jl")
        # @require OpenSpiel = "ceb70bd2-fe3f-44f0-b81f-41608acaf2f2" include("experiments/open_spiel/open_spiel.jl")
    end
end

end # module
