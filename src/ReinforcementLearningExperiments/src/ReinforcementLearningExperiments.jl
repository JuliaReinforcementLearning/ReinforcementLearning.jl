module ReinforcementLearningExperiments

using ReinforcementLearning
using Requires
using StableRNGs
using Flux
using Flux.Losses

include("experiment.jl")
include("rl_envs/rl_envs.jl")

# dynamic loading environments
function __init__()
    @require ArcadeLearningEnvironment = "b7f77d8d-088d-5e02-8ac0-89aab2acc977" include("atari/atari.jl")
    @require OpenSpiel = "ceb70bd2-fe3f-44f0-b81f-41608acaf2f2" include("open_spiel/open_spiel.jl")
    @require GridWorlds = "e15a9946-cd7f-4d03-83e2-6c30bacb0043" include("gridworlds/gridworlds.jl")
end

end # module
