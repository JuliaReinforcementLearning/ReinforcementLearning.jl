module ReinforcementLearningZoo

const RLZoo = ReinforcementLearningZoo
export RLZoo

using CircularArrayBuffers
using ReinforcementLearningBase
using ReinforcementLearningCore
using Setfield: @set
using StableRNGs
using Logging
using Flux.Losses
using Dates
using IntervalSets
using Random
using Random: shuffle
using CUDA
using Zygote
using Zygote: ignore
using Flux
using Flux: onehot, normalise
using StatsBase
using StatsBase: sample, Weights, mean
using LinearAlgebra: dot
using MacroTools
using Distributions: Categorical, Normal, logpdf
using StructArrays


include("patch.jl")
include("algorithms/algorithms.jl")

using Requires

# dynamic loading environments
function __init__()
    @require ReinforcementLearningEnvironments = "25e41dd2-4622-11e9-1641-f1adca772921" begin
        include("experiments/rl_envs/rl_envs.jl")
        @require ArcadeLearningEnvironment = "b7f77d8d-088d-5e02-8ac0-89aab2acc977" include("experiments/atari/atari.jl")
        # @require OpenSpiel = "ceb70bd2-fe3f-44f0-b81f-41608acaf2f2" include("experiments/open_spiel/open_spiel.jl")
    end
end

end # module
