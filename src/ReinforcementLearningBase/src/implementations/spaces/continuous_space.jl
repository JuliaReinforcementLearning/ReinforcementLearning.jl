export ContinuousSpace

using Random

struct ContinuousSpace{T<:Number} <: AbstractSpace
    low::T
    high::T
    function ContinuousSpace(low::T, high::T) where {T<:Number}
        low < high || throw(ArgumentError("$low must be less than $high"))
        return new{T}(low, high)
    end
end

"""
    ContinuousSpace(low, high)

Similar to [`DiscreteSpace`](@ref), but the span is continuous.
"""
ContinuousSpace(low, high) = ContinuousSpace(promote(low, high)...)

Base.eltype(::ContinuousSpace{T}) where {T} = T
Base.in(x, s::ContinuousSpace) = s.low <= x <= s.high

Random.rand(rng::AbstractRNG, s::ContinuousSpace{T}) where {T} =
    rand(rng, T) * (s.high - s.low) + s.low

Base.length(::ContinuousSpace) = throw(DomainError("ContinuousSpace is uncountable"))
