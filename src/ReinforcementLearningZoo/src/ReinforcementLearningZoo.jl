module ReinforcementLearningZoo

const RLZoo = ReinforcementLearningZoo
export RLZoo

export GaussianNetwork

using CircularArrayBuffers
using ReinforcementLearningBase
using ReinforcementLearningCore
using Setfield: @set
using Logging
using Flux.Losses
using Dates
using IntervalSets
using Random
using Random: shuffle
using CUDA
using Zygote
using Zygote: ignore, @ignore
using Flux
using Flux: onehot, normalise
using StatsBase
using StatsBase: sample, Weights, mean, std
using LinearAlgebra: dot
using MacroTools
using Distributions: Categorical, Normal, TruncatedNormal, logpdf
using StructArrays


include("patch.jl")
include("algorithms/algorithms.jl")

end # module
