module ReinforcementLearningExperiments

using Reexport
using Pkg
using Requires

@reexport using ReinforcementLearningCore, ReinforcementLearningBase, ReinforcementLearningZoo, ReinforcementLearningEnvironments

const EXPERIMENTS_DIR = joinpath(@__DIR__, "experiments")
# as long as there are not working experiments, this is not working properly
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
include(joinpath(EXPERIMENTS_DIR, "DQN_CartPoleGPU.jl"))
include(joinpath(EXPERIMENTS_DIR, "JuliaRL_SAC_Pendulum.jl"))
include(joinpath(EXPERIMENTS_DIR, "JuliaRL_CQLSAC_Pendulum.jl"))

# dynamic loading environments
function __init__()
    #PyCall environments experiments
    @require PyCall = "438e738f-606a-5dbb-bf0a-cddfbfd45ab0" include(
        joinpath(EXPERIMENTS_DIR, "pettingzoo_ex.jl")
    )
end

end # module
