export TupleSpace

const TupleSpace = Tuple{Vararg{<:AbstractSpace}}

Base.eltype(s::TupleSpace) = Tuple{eltype(x) for x in s}
Base.in(xs, ts::TupleSpace) = length(xs) == length(ts) && all(x in s for (x, s) in zip(xs, ts))
Base.rand(rng::AbstractRNG, ts::TupleSpace) = Tuple(rand(rng, s) for s in ts)