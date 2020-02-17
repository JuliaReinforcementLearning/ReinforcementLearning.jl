export WeightedExplorer

using StatsBase: sample, Weights

"""
    WeightedExplorer(;is_normalized::Bool)

`is_normalized` is used to indicate if the feeded action values
are alrady normalized to have a sum of `1.0`.
"""
struct WeightedExplorer{T} <: AbstractExplorer
    WeightedExplorer(;is_normalized::Bool=false) = new{is_normalized}()
end

(s::WeightedExplorer{true})(values::AbstractVector{T}) where {T} = sample(Weights(values, one(T)))
(s::WeightedExplorer{false})(values::AbstractVector) = sample(Weights(values, sum(values)))