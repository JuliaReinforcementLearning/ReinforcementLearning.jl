export DiscreteSpace
using Random

struct DiscreteSpace{T} <: AbstractSpace
    span::T
end

DiscreteSpace(high::T) where {T<:Integer} = DiscreteSpace(one(T), high)

function DiscreteSpace(low::T, high::T) where {T<:Integer}
    high >= low || throw(ArgumentError("$high must be >= $low"))
    DiscreteSpace(low:high)
end

DiscreteSpace(low, high) = DiscreteSpace(promote(low, high)...)

Base.eltype(s::DiscreteSpace) = eltype(s.span)
Base.in(x, s::DiscreteSpace) = x in s.span
Random.rand(rng::AbstractRNG, s::DiscreteSpace) = rand(rng, s.span)

Base.length(s::DiscreteSpace) = length(s.span)

Base.convert(::Type{AbstractSpace}, s::Union{<:Integer,<:UnitRange,<:Vector,<:Tuple}) =
    DiscreteSpace(s)

Base.iterate(s::DiscreteSpace, args...) = iterate(s.span, args...)
