export MultiContinuousSpace

using Random

struct MultiContinuousSpace{T<:AbstractArray} <: AbstractSpace
    low::T
    high::T
    function MultiContinuousSpace(low::T, high::T) where {T<:AbstractArray}
        size(low) == size(high) ||
            throw(ArgumentError("$(size(low)) != $(size(high)), size must match"))
        all(map((l, h) -> l <= h, low, high)) ||
            throw(ArgumentError("each element of $low must be ≤ than $high"))
        return new{T}(low, high)
    end
end

"""
    MultiContinuousSpace(low, high)

Similar to [`ContinuousSpace`](@ref), but scaled to multi-dimension.
"""
MultiContinuousSpace(low, high) = MultiContinuousSpace(promote(low, high)...)

Base.eltype(::MultiContinuousSpace{T}) where {T} = T
Base.in(xs, s::MultiContinuousSpace) =
    size(xs) == size(s.low) && all(map((l, x, h) -> l <= x <= h, s.low, xs, s.high))

Base.length(s::MultiContinuousSpace) = error("MultiContinuousSpace is uncountable")

function Random.rand(rng::AbstractRNG, s::MultiContinuousSpace{T}) where {T}
    return (s.high .- s.low) .* rand(rng, eltype(T), size(s.low)...) .+ s.low
end
