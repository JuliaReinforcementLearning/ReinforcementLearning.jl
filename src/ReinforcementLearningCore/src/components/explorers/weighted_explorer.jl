export WeightedExplorer

using Random
using StatsBase: sample, Weights

"""
    WeightedExplorer(;is_normalized::Bool, rng=Random.GLOBAL_RNG)

`is_normalized` is used to indicate if the feeded action values
are alrady normalized to have a sum of `1.0`.

!!! warning
    Elements are assumed to be `>=0`.

See also: [`WeightedSoftmaxExplorer`](@ref)
"""
struct WeightedExplorer{T,R<:AbstractRNG} <: AbstractExplorer
    rng::R
end

function WeightedExplorer(; is_normalized::Bool = false, rng = Random.GLOBAL_RNG)
    WeightedExplorer{is_normalized,typeof(rng)}(rng)
end

(s::WeightedExplorer{true})(values::AbstractVector{T}) where {T} =
    sample(s.rng, Weights(values, one(T)))

(s::WeightedExplorer{false})(values::AbstractVector{T}) where {T} =
    sample(s.rng, Weights(values))

function (s::WeightedExplorer)(values, mask)
    values[.!mask] .= 0
    s(values)
end
