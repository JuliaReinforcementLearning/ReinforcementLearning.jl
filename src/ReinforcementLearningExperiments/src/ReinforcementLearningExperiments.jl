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
using BSON

import ReinforcementLearning: Experiment

export @experiment_cmd, @E_cmd, Experiment

const EXPERIMENTS_DIR = joinpath(@__DIR__, "experiments")
for f in readdir(EXPERIMENTS_DIR)
    include(joinpath(EXPERIMENTS_DIR, f))
end

# dynamic loading environments
function __init__() end

end # module
