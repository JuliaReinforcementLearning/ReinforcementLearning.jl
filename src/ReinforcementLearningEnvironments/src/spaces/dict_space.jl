export DictSpace

const DictSpace = Dict{<:AbstractString, <:AbstractSpace}

Base.eltype(s::DictSpace{k}) where k = Dict{k}
Base.in(xs::Dict, ds::DictSpace) = length(xs) == length(ts) && all(haskey(ds, k) && x in ds[k] for (k, x) in xs)
Base.rand(rng::AbstractRNG, ds::DictSpace) = Dict(k => rand(rng, s) for (k, s) in ts)
