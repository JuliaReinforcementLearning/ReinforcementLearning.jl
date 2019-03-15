module ReinforcementLearningEnvironments

using Reexport, Requires

include("abstractenv.jl")
include("spaces/spaces.jl")

# built-in environments
include("environments/classic_control/classic_control.jl")

# dynamic loading environments
function __init__()
    @require ArcadeLearningEnvironment="b7f77d8d-088d-5e02-8ac0-89aab2acc977" include("environments/atari.jl")
end

end # module
