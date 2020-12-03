export WeightedSoftmaxExplorer

using Random
using StatsBase: sample, Weights
using Flux: softmax

"""
    WeightedSoftmaxExplorer(;rng=Random.GLOBAL_RNG)

See also: [`WeightedExplorer`](@ref)
"""
struct WeightedSoftmaxExplorer{R<:AbstractRNG} <: AbstractExplorer
    rng::R
end

function WeightedSoftmaxExplorer(; rng = Random.GLOBAL_RNG)
    WeightedSoftmaxExplorer(rng)
end

(s::WeightedSoftmaxExplorer)(values::AbstractVector{T}) where {T} =
    sample(s.rng, Weights(softmax(values), one(T)))

function (s::WeightedSoftmaxExplorer)(values::AbstractVector{T}, mask) where {T}
    values[.!mask] .= typemin(T)
    s(values)
end
