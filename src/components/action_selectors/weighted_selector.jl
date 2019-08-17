export WeightedSelector

using StatsBase:sample, Weights

"""
    WeightedSelector <: AbstractDiscreteActionSelector
    WeightedSelector(is_normalized::Bool)

`is_normalized` is used to indicating if the feeded action values
are alrady normalized to have a sum of `1.0`.
"""
struct WeightedSelector <: AbstractDiscreteActionSelector
    is_normalized::Bool
end

"""
    (p::WeightedSelector)(values::AbstractVector)

!!! note
    Action `values` are normalized to have a sum of 1.0
    and then used as the probability to sample a random action.
"""
(s::WeightedSelector)(values::AbstractVector{T}; kw...) where T = sample(Weights(values, s.is_normalized ? one(T) : sum(wsum)))