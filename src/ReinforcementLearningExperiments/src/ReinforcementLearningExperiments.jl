module ReinforcementLearningExperiments

using ReinforcementLearning
using Requires
using StableRNGs
using Flux
using Flux.Losses
using Setfield
using Dates
using TensorBoardLogger
using Logging
using Distributions
using IntervalSets

import ReinforcementLearning: Experiment

export @experiment_cmd, @E_cmd, Experiment

include("rl_envs/rl_envs.jl")
include("atari/atari.jl")
include("open_spiel/open_spiel.jl")
include("gridworlds/gridworlds.jl")

# dynamic loading environments
function __init__()
end

end # module
