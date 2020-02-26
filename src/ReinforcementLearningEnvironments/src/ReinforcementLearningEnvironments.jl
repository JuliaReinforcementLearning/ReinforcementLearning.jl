module ReinforcementLearningEnvironments

using ReinforcementLearningBase

export RLEnvs
const RLEnvs = ReinforcementLearningEnvironments

using Requires

# built-in environments
include("environments/non_interactive/non_interactive.jl")
include("environments/classic_control/classic_control.jl")

# dynamic loading environments
function __init__()
    @require ArcadeLearningEnvironment = "b7f77d8d-088d-5e02-8ac0-89aab2acc977" include("environments/atari.jl")
    @require PyCall = "438e738f-606a-5dbb-bf0a-cddfbfd45ab0" include("environments/gym.jl")
    @require POMDPs = "a93abf59-7444-517b-a68a-c42f96afdd7d" include("environments/mdp.jl")
    @require OpenSpiel = "ceb70bd2-fe3f-44f0-b81f-41608acaf2f2" include("environments/open_spiel.jl")
end

end # module
