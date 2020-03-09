export CircularCompactPSARTSATrajectory

using MacroTools: @forward

struct CircularCompactPSARTSATrajectory{T<:CircularCompactSARTSATrajectory,P,names,types} <:
       AbstractTrajectory{names,types}
    trajectory::T
    priority::P
end

"""
    CircularCompactPSARTSATrajectory(;kwargs)

Similar to [`CircularCompactSARTSATrajectory`](@ref), except that another trace named `priority` is added.

# Key word arguments

- `capacity::Int`, the maximum length of each trace.
- `state_type = Int`
- `state_size = ()`
- `action_type = Int`
- `action_size = ()`
- `reward_type = Float32`
- `reward_size = ()`
- `terminal_type = Bool`
- `terminal_size = ()`
- `priority_type = Float32`
"""
function CircularCompactPSARTSATrajectory(;priority_type=Float32, kw...)
    t = CircularCompactSARTSATrajectory(; kw...)
    p = SumTree(priority_type, kw.data.capacity)
    names = typeof(t).parameters[1]
    types = typeof(t).parameters[2]
    CircularCompactPSARTSATrajectory{
        typeof(t),
        typeof(p),
        (names..., :priority),
        Tuple{types.parameters...,eltype(p)},
    }(
        t,
        p,
    )
end

@forward CircularCompactPSARTSATrajectory.trajectory Base.length, Base.isempty

RLBase.get_trace(t::CircularCompactPSARTSATrajectory, s::Symbol) =
    s == :priority ? t.priority : get_trace(t.trajectory, s)

function Base.getindex(b::CircularCompactPSARTSATrajectory, i::Int)
    (
        priority = select_last_dim(b.priority, i),
        state = select_last_dim(b.trajectory[:state], i),
        action = select_last_dim(b.trajectory[:action], i),
        reward = select_last_dim(b.trajectory[:reward], i),
        terminal = select_last_dim(b.trajectory[:terminal], i),
        next_state = select_last_dim(b.trajectory[:state], i + 1),
        next_action = select_last_dim(b.trajectory[:action], i + 1),
    )
end

function Base.empty!(b::CircularCompactPSARTSATrajectory)
    empty!(b.priority)
    empty!(b.trajectory)
end

function Base.push!(b::CircularCompactPSARTSATrajectory, kv::Pair{Symbol})
    k, v = kv
    if k == :priority
        push!(b.priority, v)
    else
        push!(b.trajectory, kv)
    end
end

function Base.pop!(t::CircularCompactPSARTSATrajectory, s::Symbol)
    if s == :priority
        pop!(t.priority)
    else
        pop!(t.trajectory, s)
    end
end

function Base.pop!(t::CircularCompactPSARTSATrajectory)
    (priority=pop!(t.priority), pop!(t.trajectory)...)
end
