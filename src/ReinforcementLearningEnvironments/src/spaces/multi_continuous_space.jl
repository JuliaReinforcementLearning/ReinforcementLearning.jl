export MultiContinuousSpace
using Distributions: Uniform
using Random: AbstractRNG

struct MultiContinuousSpace{S,N} <: AbstractDiscreteSpace
    low::Array{Float64,N}
    high::Array{Float64,N}
    function MultiContinuousSpace(low::Array{Float64}, high::Array{Float64})
        size(low) == size(high) || throw(ArgumentError("$(size(low)) != $(size(high)), size must match"))
        all(l < h for (l, h) in zip(
            low,
            high,
        )) || throw(ArgumentError("each element of $low must be less than $high"))
        new{size(low),ndims(low)}(low, high)
    end
end

MultiContinuousSpace(low, high) =
    MultiContinuousSpace(convert(Array{Float64}, low), convert(Array{Float64}, high))

Base.eltype(::MultiContinuousSpace{S,N}) where {S,N} = Array{Float64,N}
Base.in(xs, s::MultiContinuousSpace{S,N}) where {S,N} =
    size(xs) == S && all(l <= x <= h for (l, x, h) in zip(s.low, xs, s.high))
Base.:(==)(s1::MultiContinuousSpace, s2::MultiContinuousSpace) =
    s1.low == s2.low && s1.high == s2.high
Base.rand(rng::AbstractRNG, s::MultiContinuousSpace) =
    map((l, h) -> rand(rng, Uniform(l, h)), s.low, s.high)
Base.size(s::MultiContinuousSpace) = size(s.low)
Base.length(s::MultiContinuousSpace) = length(s.low)