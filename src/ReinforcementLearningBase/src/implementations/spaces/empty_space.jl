export EmptySpace

"""
    EmptySpace()

There's `nothing` in the `EmptySpace`!
"""
struct EmptySpace <: AbstractSpace end

Base.eltype(s::EmptySpace) = Nothing
Base.length(s::EmptySpace) = 0
Base.in(x, s::EmptySpace) = x isa Nothing
Random.rand(rng::AbstractRNG, s::EmptySpace) = nothing
