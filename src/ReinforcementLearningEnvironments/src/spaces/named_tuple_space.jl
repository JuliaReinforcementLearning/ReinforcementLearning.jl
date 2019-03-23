export NamedTupleSpace

const NamedTupleSpace = NamedTuple{name, <:TupleSpace} where name

Base.eltype(s::NamedTupleSpace{name}) where name = NamedTuple{name, Tuple{eltype(x) for x in s}}
Base.in(xs, nts::NamedTupleSpace) = length(x) == length(s) && all(x in s for (x, s) in zip(xs, nts))
Base.rand(rng::AbstractRNG, nts::NamedTupleSpace{name}) where name = NamedTuple{name}(rand(rng, s) for s in nts)