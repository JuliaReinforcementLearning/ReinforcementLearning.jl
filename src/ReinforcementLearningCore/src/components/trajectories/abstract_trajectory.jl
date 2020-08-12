export AbstractTrajectory

"""
    AbstractTrajectory

A trace is used to record some useful information
during the interactions between agents and environments.

Required Methods:

- `Base.haskey(t::AbstractTrajectory, s::Symbol)`
- `Base.getindex(t::AbstractTrajectory, s::Symbol)`
- `Base.keys(t::AbstractTrajectory)`
- `Base.push!(t::AbstractTrajectory, kv::Pair{Symbol})`
- `Base.pop!(t::AbstractTrajectory, s::Symbol)`
- `Base.empty!(t::AbstractTrajectory)`

Optional Methods:

- `isfull`

"""
abstract type AbstractTrajectory end

function Base.push!(t::AbstractTrajectory;kwargs...)
    for kv in kwargs
        push!(t, kv)
    end
end

"""
    Base.pop!(t::AbstractTrajectory, s::Symbol...)

`pop!` out one element of the traces specified in `s`
"""
function Base.pop!(t::AbstractTrajectory, s::Tuple{Vararg{Symbol}})
    NamedTuple{s}(pop!(t, x) for x in s)
end

Base.pop!(t::AbstractTrajectory) = pop!(t, keys(t))

function Base.empty!(t::AbstractTrajectory)
    for s in keys(t)
        empty!(t[s])
    end
end

#####
# patch code
#####

# avoid showing the inner structure
function AbstractTrees.children(t::StructTree{<:AbstractTrajectory})
    Tuple(k => StructTree(t.x[k]) for k in keys(t.x))
end

@deprecate get_trace(t::AbstractTrajectory, s::Symbol) t[s]