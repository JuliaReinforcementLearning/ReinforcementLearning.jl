import Base:occursin, size, ==
export AbstractSpace,
       BoxSpace,
       DiscreteSpace,
       MultiBinarySpace,
       MultiDiscreteSpace,
       sample

include("abstractspace.jl")
include("boxspace.jl")
include("discretespace.jl")
include("multibinaryspace.jl")
include("multidiscretespace.jl")

# Tuple Support
sample(s::Tuple{Vararg{<:AbstractSpace}}) = map(sample, s)
occursin(a::Tuple, b::Tuple{Vararg{<:AbstractSpace}}) = length(a) == length(b) &&
    all(map((x, y) -> occursin(x, y), a, b))

# Dict Support
sample(s::Dict{String}) = Dict(map((k, v) -> (k, sample(v)), s))
occursin(a::Dict{String}, b::Dict{String}) = length(a) == length(b) &&
    all(p -> haskey(a, p.first) ? 
            occursin(a[p.first], p.second) :
            false,
        b)