export EmptySpace

struct EmptySpace <: AbstractSpace end

Base.eltype(s::EmptySpace) = nothing
Base.length(s::EmptySpace) = 0
Base.in(x, s::EmptySpace) = false
Random.rand(rng::AbstractRNG, s::EmptySpace) = nothing
