export DictSpace

struct DictSpace{K<:Union{Symbol,AbstractString},V<:AbstractSpace} <: AbstractSpace
    data::Dict{K,V}
end

"""
    DictSpace(ps::Pair{<:Union{Symbol,AbstractString},<:AbstractSpace}...)
"""
function DictSpace(ps::Pair{<:Union{Symbol,AbstractString},<:AbstractSpace}...)
    data = Dict(ps)
    K, V = typeof(data).parameters
    return DictSpace{K,V}(Dict(ps))
end

Base.eltype(::DictSpace{K}) where {K} = Dict{K}
function Base.in(xs::Dict, s::DictSpace)
    return length(xs) == length(s.data) &&
           all(haskey(s.data, k) && x in s.data[k] for (k, x) in xs)
end
Random.rand(rng::AbstractRNG, s::DictSpace) = Dict(k => rand(rng, s) for (k, s) in s.data)
