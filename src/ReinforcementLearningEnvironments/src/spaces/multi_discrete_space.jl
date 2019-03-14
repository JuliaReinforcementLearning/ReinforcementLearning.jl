export MultiDiscreteSpace
using Random:AbstractRNG

struct MultiDiscreteSpace{N} <:AbstractSpace
    counts::Array{Int, N}
    function MultiDiscreteSpace(xs::Array{Int, N}) where {N}
        all(x > 0 for x in xs) || throw(ArgumentError("each element of $xs must be greater than zero"))
        new{N}(convert(Array{Int}, xs))
    end
end

MultiDiscreteSpace(xs) = MultiDiscreteSpace(convert(Array{Int}, xs))

Base.length(s::MultiDiscreteSpace) = *(s.counts...)
Base.eltype(s::MultiDiscreteSpace{N}) where N = Array{Int, N}
Base.in(xs, s::MultiDiscreteSpace) = length(xs) == length(s.counts) && all(1 <= convert(Int, x) <= c for (x, c) in zip(xs, s.counts))
Base. ==(s1::MultiDiscreteSpace, s2::MultiDiscreteSpace) = s1.counts == s2.counts
Base.rand(rng::AbstractRNG, s::MultiDiscreteSpace) = map(c -> rand(rng, 1:c), s.counts)