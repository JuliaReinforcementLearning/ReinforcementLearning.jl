export TupleSpace

struct TupleSpace{T} <: AbstractSpace
    data::T
end

Base.eltype(s::TupleSpace) = Tuple{(eltype(x) for x in s.data)...}
Base.in(xs, s::TupleSpace) =
    length(xs) == length(s.data) && all(x in d for (x, d) in zip(xs, s.data))
Random.rand(rng::AbstractRNG, s::TupleSpace) = Tuple(rand(rng, d) for d in s.data)
