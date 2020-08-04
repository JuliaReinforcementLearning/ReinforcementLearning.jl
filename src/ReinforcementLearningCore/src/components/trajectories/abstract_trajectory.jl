export AbstractTrajectory, get_trace, RTSA, SARTSA

"""
    AbstractTrajectory{names,types} <: AbstractArray{NamedTuple{names,types},1}

A trajectory is used to record some useful information
during the interactions between agents and environments.

# Parameters
- `names`::`NTuple{Symbol}`, indicate what fields to be recorded.
- `types`::`Tuple{DataType...}`, the datatypes of `names`.

The length of `names` and `types` must match.

Required Methods:

- [`get_trace`](@ref)
- `Base.push!(t::AbstractTrajectory, kv::Pair{Symbol})`
- `Base.pop!(t::AbstractTrajectory, s::Symbol)`

Optional Methods:

- `Base.length`
- `Base.size`
- `Base.lastindex`
- `Base.isempty`
- `Base.empty!`
"""
abstract type AbstractTrajectory{names,types} <: AbstractArray{NamedTuple{names,types},1} end

# some typical trace names
"An alias of `(:reward, :terminal, :state, :action)`"
const RTSA = (:reward, :terminal, :state, :action)

"An alias of `(:state, :action, :reward, :terminal, :next_state, :next_action)`"
const SARTSA = (:state, :action, :reward, :terminal, :next_state, :next_action)

"""
    get_trace(t::AbstractTrajectory, s::NTuple{N,Symbol}) where {N}
"""
get_trace(t::AbstractTrajectory, s::NTuple{N,Symbol}) where {N} =
    NamedTuple{s}(get_trace(t, x) for x in s)

"""
    get_trace(t::AbstractTrajectory, s::Symbol...)
"""
get_trace(t::AbstractTrajectory, s::Symbol...) = get_trace(t, s)

"""
    get_trace(t::AbstractTrajectory{names}) where {names}
"""
get_trace(t::AbstractTrajectory{names}) where {names} =
    NamedTuple{names}(get_trace(t, x) for x in names)

Base.length(t::AbstractTrajectory) = maximum(length(x) for x in get_trace(t))
Base.size(t::AbstractTrajectory) = (length(t),)
Base.lastindex(t::AbstractTrajectory) = length(t)
Base.getindex(t::AbstractTrajectory{names,types}, i::Int) where {names,types} =
    NamedTuple{names,types}(Tuple(x[i] for x in get_trace(t)))

Base.isempty(t::AbstractTrajectory) = all(isempty(t) for t in get_trace(t))

function Base.empty!(t::AbstractTrajectory)
    for x in get_trace(t)
        empty!(x)
    end
end

"""
    Base.push!(t::AbstractTrajectory; kwargs...)
"""
function Base.push!(t::AbstractTrajectory; kwargs...)
    for kv in kwargs
        push!(t, kv)
    end
end

"""
    Base.pop!(t::AbstractTrajectory{names}) where {names}
`pop!` out one element of each trace in `t`
"""
function Base.pop!(t::AbstractTrajectory{names}) where {names}
    pop!(t, names...)
end

"""
    Base.pop!(t::AbstractTrajectory, s::Symbol...)
`pop!` out one element of the traces specified in `s`
"""
function Base.pop!(t::AbstractTrajectory, s::Symbol...)
    NamedTuple{s}(pop!(t, x) for x in s)
end

function AbstractTrees.children(t::StructTree{<:AbstractTrajectory})
    traces = get_trace(t.x)
    Tuple(k => StructTree(v) for (k, v) in pairs(traces))
end

Base.summary(io::IO, t::T) where {T<:AbstractTrajectory} =
    print(io, "$(length(t))-element $(T.name)")
