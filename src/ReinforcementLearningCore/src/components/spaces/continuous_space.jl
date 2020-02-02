export ContinuousSpace

using Distributions: Uniform
using Random

struct ContinuousSpace{T<:Number} <: AbstractSpace
    low::T
    high::T
    function ContinuousSpace(low::T, high::T) where {T<:Number}
        low < high || throw(ArgumentError("$low must be less than $high"))
        new{T}(low, high)
    end
end

ContinuousSpace(low, high) = ContinuousSpace(promote(low, high)...)

Base.eltype(::ContinuousSpace{T}) where {T} = T
Base.in(x, s::ContinuousSpace) = s.low <= x <= s.high

# watch https://github.com/JuliaStats/Distributions.jl/pull/951
Random.rand(rng::AbstractRNG, s::ContinuousSpace{T}) where {T} =
    convert(T, rand(rng, Uniform(s.low, s.high)))

Base.length(::ContinuousSpace) = throw(DomainError("ContinuousSpace is uncountable"))
element_length(::ContinuousSpace) = 0
element_size(::ContinuousSpace) = ()
