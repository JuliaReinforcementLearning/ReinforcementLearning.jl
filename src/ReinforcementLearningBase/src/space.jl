export ArrayProductDomain

import DomainSets
import Base.==

@reexport using DomainSets: ×, (..), fullspace, TupleProductDomain

struct ArrayProductDomain{A<:AbstractArray,DD<:AbstractArray} <: DomainSets.ProductDomain{A}
    domains::DD

    function ArrayProductDomain{A,DD}(domains::DD) where {A,DD}
        @assert eltype(eltype(domains)) == eltype(A)
        new(domains)
    end
end

ArrayProductDomain(domains::AbstractArray) =
    ArrayProductDomain{Array{eltype(eltype(domains))}}(domains)

ArrayProductDomain{V}(domains::AbstractArray{<:DomainSets.Domain{T}}) where {T,V<:AbstractArray{T}} =
    ArrayProductDomain{V,typeof(domains)}(domains)
function ArrayProductDomain{V}(domains::AbstractArray) where {T,V<:AbstractArray{T}}
    Tdomains = convert.(DomainSets.Domain{T}, domains)
    ArrayProductDomain{V}(Tdomains)
end

Base.size(d::ArrayProductDomain) = size(d.domains)

DomainSets.tointernalpoint(d::ArrayProductDomain, x) =
    (@assert size(x) == size(d); x)
DomainSets.toexternalpoint(d::ArrayProductDomain, y) =
    (@assert size(y) == size(d); y)

DomainSets.promote_pair(x::AbstractArray, d::DomainSets.Domain{<:AbstractArray}) = x, d

# NOTE: This is a patch to maintain current behavior while using DomainSets >=0.7
function ==(a1::ArrayProductDomain, a2::ArrayProductDomain)
    try
        return isequaldomain(a1, a2)
    catch e
        @warn "Use StaticArrays (e.g. SVector, SMatrix, or SArray when constructing `ArrayProductDomain` objects."
        return typeof(a1) == typeof(a2)
    end
end

#####
export NamedTupleProductDomain

struct NamedTupleProductDomain{T,D,N} <: DomainSets.ProductDomain{T}
    tuple_product_domain::TupleProductDomain{T,D}
    name2ind::N
end

function NamedTupleProductDomain(; kv...)
    t = TupleProductDomain(values(kv)...)
    name2ind = NamedTuple(k => i for (i, k) in enumerate(keys(kv)))
    NamedTupleProductDomain(t, name2ind)
end

# NOTE: This is a patch to maintain current behavior while using DomainSets >=0.7
function ==(a1::NamedTupleProductDomain, a2::NamedTupleProductDomain)
    try
        return isequaldomain(a1, a2)
    catch e
        @warn "Use StaticArrays (e.g. SVector, SMatrix, or SArray when constructing `NamedTupleProductDomain` objects."
        return typeof(a1) == typeof(a2)
    end
end

DomainSets.components(d::NamedTupleProductDomain) = components(d.tuple_product_domain)
Base.getindex(d::NamedTupleProductDomain, x::Symbol) = components(d)[d.name2ind[x]]
Base.getindex(d::NamedTupleProductDomain, x::Int) = components(d)[x]
Base.getindex(d::TupleProductDomain, x::Int) = components(d)[x]

DomainSets.tointernalpoint(d::TupleProductDomain, x) =
    (@assert length(x) == length(d.domains); x)
DomainSets.toexternalpoint(d::TupleProductDomain, y) =
    (@assert length(y) == length(d.domains); y)
DomainSets.tointernalpoint(d::NamedTupleProductDomain, x) =
    (@assert length(x) == length(d.domains); x)
DomainSets.toexternalpoint(d::NamedTupleProductDomain, y) =
    (@assert length(y) == length(d.domains); y)
