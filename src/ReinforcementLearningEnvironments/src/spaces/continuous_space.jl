export ContinuousSpace
using Distributions:Uniform
using Random:AbstractRNG

struct ContinuousSpace <: AbstractDiscreteSpace
    low::Float64
    high::Float64
    function ContinuousSpace(low::Float64, high::Float64)
        low < high || throw(ArgumentError("$low must be less than $high"))
        new(low, high)
    end
end
ContinuousSpace(low, high) = ContinuousSpace(convert(Float64, low), convert(Float64, high))

Base.eltype(::ContinuousSpace) = Float64  # rand(Uniform(a, b)) always return Float64
Base.in(x, s::ContinuousSpace) = s.low <= x <= s.high
Base.:(==)(s1::ContinuousSpace, s2::ContinuousSpace) = s1.low == s2.low && s1.high == s2.high
Base.rand(rng::AbstractRNG, s::ContinuousSpace) = rand(rng, Uniform(s.low, s.high))