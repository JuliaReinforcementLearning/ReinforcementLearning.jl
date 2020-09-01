module ReinforcementLearningZoo

const RLZoo = ReinforcementLearningZoo
export RLZoo

using ReinforcementLearningBase
using ReinforcementLearningCore
using Setfield: @set

include("patch.jl")
include("algorithms/algorithms.jl")
include("utils.jl")

using Requires

# dynamic loading environments
function __init__()
    @require ReinforcementLearningEnvironments = "25e41dd2-4622-11e9-1641-f1adca772921" begin
        include("experiments/rl_envs.jl")
        @require ArcadeLearningEnvironment = "b7f77d8d-088d-5e02-8ac0-89aab2acc977" include("experiments/atari.jl")
        @require SnakeGames = "34dccd9f-48d6-4445-aa0f-8c2e373b5429" include("experiments/snake.jl")
        @require OpenSpiel = "ceb70bd2-fe3f-44f0-b81f-41608acaf2f2" include("experiments/open_spiel.jl")
    end
end

end # module
