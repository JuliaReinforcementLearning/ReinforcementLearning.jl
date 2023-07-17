module ReinforcementLearningExperiments

using Reexport
using Requires

@reexport using ReinforcementLearningCore, ReinforcementLearningBase, ReinforcementLearningZoo

const EXPERIMENTS_DIR = joinpath(@__DIR__, "experiments")
# for f in readdir(EXPERIMENTS_DIR)
#     include(joinpath(EXPERIMENTS_DIR, f))
# end
include(joinpath(EXPERIMENTS_DIR, "JuliaRL_NFQ_CartPole.jl"))
include(joinpath(EXPERIMENTS_DIR, "JuliaRL_BasicDQN_CartPole.jl"))
include(joinpath(EXPERIMENTS_DIR, "JuliaRL_DQN_CartPole.jl"))
include(joinpath(EXPERIMENTS_DIR, "JuliaRL_PrioritizedDQN_CartPole.jl"))
include(joinpath(EXPERIMENTS_DIR, "JuliaRL_QRDQN_CartPole.jl"))
include(joinpath(EXPERIMENTS_DIR, "JuliaRL_REMDQN_CartPole.jl"))
include(joinpath(EXPERIMENTS_DIR, "JuliaRL_IQN_CartPole.jl"))
include(joinpath(EXPERIMENTS_DIR, "JuliaRL_Rainbow_CartPole.jl"))
include(joinpath(EXPERIMENTS_DIR, "JuliaRL_VPG_CartPole.jl"))
include(joinpath(EXPERIMENTS_DIR, "JuliaRL_TRPO_CartPole.jl"))
include(joinpath(EXPERIMENTS_DIR, "JuliaRL_MPO_CartPole.jl"))
include(joinpath(EXPERIMENTS_DIR, "IDQN_TicTacToe.jl"))


# dynamic loading environments
function __init__() 
    @require PyCall = "438e738f-606a-5dbb-bf0a-cddfbfd45ab0" include(
        joinpath(EXPERIMENTS_DIR, "DQN_mpe_simple.jl")
    )
end

end # module
