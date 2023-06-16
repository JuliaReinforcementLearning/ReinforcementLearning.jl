export WeightedSoftmaxExplorer

using Random: AbstractRNG
using StatsBase: sample, Weights
using Flux: softmax

"""
    WeightedSoftmaxExplorer(;rng=Random.default_rng())

See also: [`WeightedExplorer`](@ref)
"""
struct WeightedSoftmaxExplorer <: AbstractExplorer
    rng::AbstractRNG
end

function WeightedSoftmaxExplorer(; rng = Random.default_rng())
    WeightedSoftmaxExplorer(rng)
end

RLBase.plan!(s::WeightedSoftmaxExplorer, values::AbstractVector{T}) where {T} =
    sample(s.rng, Weights(softmax(values), one(T)))

function RLBase.plan!(s::WeightedSoftmaxExplorer, values::AbstractVector{T}, mask) where {T}
    values[.!mask] .= typemin(T)
    s(values)
end

RLBase.prob(s::WeightedSoftmaxExplorer, values) = softmax(values)

function RLBase.prob(s::WeightedSoftmaxExplorer, values::AbstractVector{T}, mask) where {T}
    p = prob(s, values) .* mask
    p ./= sum(p)
    p
end
