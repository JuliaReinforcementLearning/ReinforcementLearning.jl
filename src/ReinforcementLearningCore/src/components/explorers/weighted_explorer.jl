export WeightedExplorer

using Random
using StatsBase: sample, Weights
using Flux: softmax

"""
    WeightedExplorer(;is_normalized::Bool)

`is_normalized` is used to indicate if the feeded action values
are alrady normalized to have a sum of `1.0`.
"""
struct WeightedExplorer{T,R<:AbstractRNG} <: AbstractExplorer
    rng::R
end

function WeightedExplorer(; is_normalized::Bool = false, rng = Random.GLOBAL_RNG)
    WeightedExplorer{is_normalized,typeof(rng)}(rng)
end

(s::WeightedExplorer{true})(values::AbstractVector{T}) where {T} =
    sample(s.rng, Weights(values, one(T)))

# ??? add a softmax layer here?
(s::WeightedExplorer{false})(values::AbstractVector{T}) where {T} =
    sample(s.rng, Weights(softmax(values), one(T)))
