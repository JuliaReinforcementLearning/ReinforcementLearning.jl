export VectSpace

struct VectSpace{T} <: AbstractSpace
    data::T
end

Base.eltype(s::VectSpace) = Vector{eltype(s.data[1])}
Base.in(xs, s::VectSpace) =
    length(xs) == length(s.data) && all(x in d for (x, d) in zip(xs, s.data))
Random.rand(rng::AbstractRNG, s::VectSpace) = [rand(rng, d) for d in s.data]

Base.iterate(s::VectSpace, args...) = iterate(s.data, args...)
Base.getindex(s::VectSpace, i::Int) = getindex(s.data, i)
Base.length(s::VectSpace) = length(s.data)

"""
    getindex(s::VectSpace, I::Vector{Int})

Here `I` represents the index of action in each inner space inside `s`.
"""
function Base.getindex(s::VectSpace, I::Vector{Int})
    @assert length(s.data) == length(I)
    return [getindex(d, i) for (d, i) in zip(s.data, I)]
end
