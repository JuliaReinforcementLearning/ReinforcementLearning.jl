export VectSpace

struct VectSpace{T} <: AbstractSpace
    data::T
end

Base.eltype(s::VectSpace) = Vector{eltype(s.data[1])}
Base.in(xs, s::VectSpace) =
    length(xs) == length(s.data) && all(x in d for (x, d) in zip(xs, s.data))
Random.rand(rng::AbstractRNG, s::VectSpace) = [rand(rng, d) for d in s.data]
