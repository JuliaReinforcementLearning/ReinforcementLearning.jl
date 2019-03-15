module ReinforcementLearningEnvironments

using Reexport

include("abstractenv.jl")
include("spaces/spaces.jl")

# built-in environments
include("environments/classic_control/classic_control.jl")

# dynamic loading environments


end # module
