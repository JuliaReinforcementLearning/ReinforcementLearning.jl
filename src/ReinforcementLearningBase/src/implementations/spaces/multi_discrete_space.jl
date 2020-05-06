export MultiDiscreteSpace
using Random

"""
    MultiDiscreteSpace(low::T, high::T) where {T<:AbstractArray}

Similar to [`DiscreteSpace`](@ref), but scaled to multi-dimension.
"""
struct MultiDiscreteSpace{T<:AbstractArray} <: AbstractSpace
    low::T
    high::T
    n::Int  # pre-calculation
    function MultiDiscreteSpace(low::T, high::T) where {T<:AbstractArray}
        all(map((l, h) -> l <= h, low, high)) ||
            throw(ArgumentError("each element of $high must be ≥r $low"))
        new{T}(low, high, reduce(*, map((l, h) -> h - l + 1, low, high)))
    end
end

"""
    MultiDiscreteSpace(high::T) where {T<:AbstractArray}

The `low` will fall back to `ones(eltype(T), size(high))`.
"""
MultiDiscreteSpace(high::T) where {T<:AbstractArray} =
    MultiDiscreteSpace(ones(eltype(T), size(high)), high)

Base.length(s::MultiDiscreteSpace) = s.n
Base.eltype(s::MultiDiscreteSpace{T}) where {T} = T
Base.in(xs, s::MultiDiscreteSpace) =
    size(xs) == size(s.low) && all(map((l, x, h) -> l <= x <= h, s.low, xs, s.high))
Random.rand(rng::AbstractRNG, s::MultiDiscreteSpace) =
    map((l, h) -> rand(rng, l:h), s.low, s.high)
