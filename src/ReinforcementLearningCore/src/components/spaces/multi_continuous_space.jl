export MultiContinuousSpace
using Distributions: Uniform
using Random: AbstractRNG

struct MultiContinuousSpace{N} <: AbstractSpace
    low::Array{Float64,N}
    high::Array{Float64,N}
    function MultiContinuousSpace(low::Array{Float64}, high::Array{Float64})
        size(low) == size(high) || throw(ArgumentError("$(size(low)) != $(size(high)), size must match"))
        all(l < h for (l, h) in zip(
            low,
            high,
        )) || throw(ArgumentError("each element of $low must be less than $high"))
        new{ndims(low)}(low, high)
    end
end

MultiContinuousSpace(low, high) =
    MultiContinuousSpace(convert(Array{Float64}, low), convert(Array{Float64}, high))

Base.eltype(::MultiContinuousSpace{N}) where {N} = Array{Float64,N}
Base.in(xs, s::MultiContinuousSpace{N}) where {N} = size(xs) == element_size(s) && all(l <= x <= h for (l, x, h) in zip(s.low, xs, s.high))
Base.rand(rng::AbstractRNG, s::MultiContinuousSpace) = map((l, h) -> rand(rng, Uniform(l, h)), s.low, s.high)

Base.length(s::MultiContinuousSpace) = error("MultiContinuousSpace is uncountable")
element_size(s::MultiContinuousSpace) = size(s.low)
element_length(s::MultiContinuousSpace) = length(s.low)