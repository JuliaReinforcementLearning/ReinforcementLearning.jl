module ReinforcementLearningEnvironments

using Reexport, Requires

include("abstractenv.jl")
include("spaces/spaces.jl")

# built-in environments
include("environments/classic_control/classic_control.jl")

# dynamic loading environments
function __init__()
    @require ArcadeLearningEnvironment = "b7f77d8d-088d-5e02-8ac0-89aab2acc977" include("environments/atari.jl")
    @require ViZDoom                   = "13bb3beb-38fe-5ca7-9a46-050a216300b1" include("environments/vizdoom.jl")
    @require PyCall                    = "438e738f-606a-5dbb-bf0a-cddfbfd45ab0" include("environments/gym.jl")
    @require Hanabi                    = "705708ad-e62c-5f47-9095-732127600058" include("environments/hanabi.jl")
end

end # module
