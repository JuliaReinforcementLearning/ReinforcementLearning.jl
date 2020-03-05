export WeightedExplorer

using Random
using StatsBase: sample, Weights

"""
    WeightedExplorer(;is_normalized::Bool)

`is_normalized` is used to indicate if the feeded action values
are alrady normalized to have a sum of `1.0`.
"""
struct WeightedExplorer{T,R<:AbstractRNG} <: AbstractExplorer
    rng::R
end

function WeightedExplorer(; is_normalized::Bool = false, seed = nothing)
    rng = MersenneTwister(seed)
    WeightedExplorer{is_normalized,typeof(rng)}(rng)
end

(s::WeightedExplorer{true})(values::AbstractVector{T}) where {T} =
    sample(s.rng, Weights(values, one(T)))
(s::WeightedExplorer{false})(values::AbstractVector) =
    sample(s.rng, Weights(values, sum(values)))
