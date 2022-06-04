module ReinforcementLearningExperiments


const EXPERIMENTS_DIR = joinpath(@__DIR__, "experiments")
# for f in readdir(EXPERIMENTS_DIR)
#     include(joinpath(EXPERIMENTS_DIR, f))
# end
include(joinpath(EXPERIMENTS_DIR, "JuliaRL_BasicDQN_CartPole.jl"))

# dynamic loading environments
function __init__() end

end # module
