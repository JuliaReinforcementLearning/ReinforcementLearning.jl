export AbstractTrajectory, SART, SARTS, SARTSA, SLART, SLARTSL, SLARTSLA

"""
    AbstractTrajectory

A trajectory is used to record some useful information
during the interactions between agents and environments.
It behaves similar to a `NamedTuple` except that we extend it
with some optional methods.

Required Methods:

- `Base.getindex`
- `Base.keys`

Optional Methods:

- `Base.length`
- `Base.isempty`
- `Base.empty!`
- `Base.haskey`
- `Base.push!`
- `Base.pop!`
"""
abstract type AbstractTrajectory end

Base.haskey(t::AbstractTrajectory, s::Symbol) = s in keys(t)
Base.isempty(t::AbstractTrajectory) = all(k -> isempty(t[k]), keys(t))

function Base.empty!(t::AbstractTrajectory)
    for k in keys(t)
        empty!(t[k])
    end
end

function Base.push!(t::AbstractTrajectory; kwargs...)
    for (k, v) in kwargs
        push!(t[k], v)
    end
end

function Base.pop!(t::AbstractTrajectory)
    for k in keys(t)
        pop!(t[k])
    end
end

function Base.show(io::IO, t::AbstractTrajectory)
    println(io, "Trajectory of $(length(keys(t))) traces:")
    for k in keys(t)
        show(io, k)
        println(io, " $(summary(t[k]))")
    end
end

#####
# Common Keys
#####

const SART = (:state, :action, :reward, :terminal)
const SARTS = (:state, :action, :reward, :terminal, :next_state)
const SARTSA = (:state, :action, :reward, :terminal, :next_state, :next_action)
const SLART = (:state, :legal_actions_mask, :action, :reward, :terminal)
const SLARTSL = (
    :state,
    :legal_actions_mask,
    :action,
    :reward,
    :terminal,
    :next_state,
    :next_legal_actions_mask,
)
const SLARTSLA = (
    :state,
    :legal_actions_mask,
    :action,
    :reward,
    :terminal,
    :next_state,
    :next_legal_actions_mask,
    :next_action,
)
