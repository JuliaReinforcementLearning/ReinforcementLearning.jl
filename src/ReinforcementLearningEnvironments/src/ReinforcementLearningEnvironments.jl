module ReinforcementLearningEnvironments

using ReinforcementLearningBase
using Random
using Requires
using IntervalSets
using Base.Threads: @spawn
using Markdown

const RLEnvs = ReinforcementLearningEnvironments
export RLEnvs

include("base.jl")
include("environments/environments.jl")
include("converters.jl")

# dynamic loading environments
function __init__()
    @require ArcadeLearningEnvironment = "b7f77d8d-088d-5e02-8ac0-89aab2acc977" include(
        "environments/3rd_party/atari.jl",
    )
    @require PyCall = "438e738f-606a-5dbb-bf0a-cddfbfd45ab0" include(
        "environments/3rd_party/gym.jl",
    )
    @require OpenSpiel = "ceb70bd2-fe3f-44f0-b81f-41608acaf2f2" include(
        "environments/3rd_party/open_spiel.jl",
    )
    @require SnakeGames = "34dccd9f-48d6-4445-aa0f-8c2e373b5429" include(
        "environments/3rd_party/snake.jl",
    )
    @require OrdinaryDiffEq = "1dea7af3-3e70-54e6-95c3-0bf5283fa5ed" include(
        "environments/3rd_party/AcrobotEnv.jl",
    )
    @require Plots = "91a5bcdd-55d7-5caf-9e0b-520d859cae80" include("plots.jl")


end

end # module
