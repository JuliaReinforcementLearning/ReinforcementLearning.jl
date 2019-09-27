export MultiDiscreteSpace
using Random: AbstractRNG

struct MultiDiscreteSpace{T<:Integer,N} <: AbstractSpace
    low::Array{T,N}
    high::Array{T,N}
    n::Int
    function MultiDiscreteSpace(
        high::Array{T,N},
        low = ones(T, size(high)),
    ) where {T<:Integer,N}
        all(l < h for (l, h) in zip(
            low,
            high,
        )) || throw(ArgumentError("each element of $high must be greater than $low"))
        new{T,N}(low, high, reduce(*, h - l + 1 for (l, h) in zip(low, high)))
    end
end

MultiDiscreteSpace(xs) = MultiDiscreteSpace(convert(Array{Int}, xs))

Base.length(s::MultiDiscreteSpace) = s.n
Base.eltype(s::MultiDiscreteSpace{T,N}) where {T,N} = Array{T,N}
Base.in(xs, s::MultiDiscreteSpace) =
    size(xs) == size(s.low) && all(l <= x <= h for (l, x, h) in zip(s.low, xs, s.high))
Base.:(==)(s1::MultiDiscreteSpace, s2::MultiDiscreteSpace) =
    s1.low == s2.low && s1.high == s2.high
Base.rand(rng::AbstractRNG, s::MultiDiscreteSpace) =
    map((l, h) -> rand(rng, l:h), s.low, s.high)