export DiscreteSpace
using Random: AbstractRNG

struct DiscreteSpace{T<:Integer} <: AbstractSpace
    low::T
    high::T
    n::T
    function DiscreteSpace(high::T, low = one(T)) where {T<:Integer}
        high >= low || throw(ArgumentError("$high must be >= $low"))
        new{T}(low, high, high - low + 1)
    end
end


Base.eltype(s::DiscreteSpace{T}) where {T} = T
Base.in(x, s::DiscreteSpace{T}) where {T} = s.low <= x <= s.high
Base.rand(rng::AbstractRNG, s::DiscreteSpace) = rand(rng, s.low:s.high)

Base.length(s::DiscreteSpace) = s.n
element_length(::DiscreteSpace) = 0
element_size(::DiscreteSpace) = ()