export Trajectory,
    SharedTrajectory,
    EpisodicTrajectory,
    CombinedTrajectory,
    CircularCompactSATrajectory,
    CircularCompactSALTrajectory,
    CircularCompactSARTSATrajectory,
    CircularCompactPSARTSATrajectory,
    CircularCompactSALRTSALTrajectory,
    CircularCompactPSALRTSALTrajectory

using MacroTools: @forward

#####
# Trajectory
#####

"""
    Trajectory(;[trace_name=trace_container]...)

Simply a wrapper of `NamedTuple`.
Define our own type here to avoid type piracy with `NamedTuple`
"""
struct Trajectory{T} <: AbstractTrajectory
    traces::T
end

Trajectory(; kwargs...) = Trajectory(kwargs.data)

const DUMMY_TRAJECTORY = Trajectory()
const DummyTrajectory = typeof(DUMMY_TRAJECTORY)

@forward Trajectory.traces Base.keys, Base.haskey, Base.getindex

Base.push!(t::Trajectory, kv::Pair{Symbol}) = push!(t[first(kv)], last(kv))
Base.pop!(t::Trajectory, s::Symbol) = pop!(t[s])

isfull(t::Trajectory) = all(isfull, t.traces)

#####
# SharedTrajectory
#####

struct SharedTrajectoryMeta
    start_shift::Int
    end_shift::Int
end

"""
    SharedTrajectory(trace_container, meta::NamedTuple{([trace_name::Symbol],...), Tuple{[SharedTrajectoryMeta]...}})

Create multiple traces sharing the same underlying container.
"""
struct SharedTrajectory{X,M} <: AbstractTrajectory
    x::X
    meta::M
end

"""
    SharedTrajectory(trace_container, s::Symbol)

Automatically create the following three traces:

- `s`, share the data in `trace_container` in the range of `1:end-1`
- `s` with a prefix of `next_`, share the data in `trace_container` in the range of `2:end`
- `s` with a prefix of `full_`, a view of `trace_container`
"""
function SharedTrajectory(x, s::Symbol)
    SharedTrajectory(
        x,
        (;
            s => SharedTrajectoryMeta(1, -1),
            Symbol(:next_, s) => SharedTrajectoryMeta(2, 0),
            Symbol(:full_, s) => SharedTrajectoryMeta(1, 0),
        ),
    )
end

@forward SharedTrajectory.meta Base.keys, Base.haskey

function Base.getindex(t::SharedTrajectory, s::Symbol)
    m = t.meta[s]
    select_last_dim(t.x, m.start_shift:(nframes(t.x)+m.end_shift))
end

Base.push!(t::SharedTrajectory, kv::Pair{Symbol}) = push!(t.x, last(kv))
Base.empty!(t::SharedTrajectory) = empty!(t.x)
Base.pop!(t::SharedTrajectory, s::Symbol) = pop!(t.x)

function Base.pop!(t::SharedTrajectory)
    s = first(keys(t))
    (; s => pop!(t.x))
end

isfull(t::SharedTrajectory) = isfull(t.x)

#####
# EpisodicTrajectory
#####

"""
    EpisodicTrajectory(traces::T, flag_trace=:terminal)

Assuming that the `flag_trace` is in `traces` and it's an `AbstractVector{Bool}`, 
meaning whether an environment reaches terminal or not. The last element in
`flag_trace` will be used to determine whether the whole trace is full or not.
"""
struct EpisodicTrajectory{T,flag_trace} <: AbstractTrajectory
    traces::T
end

EpisodicTrajectory(traces::T, flag_trace = :terminal) where {T} =
    EpisodicTrajectory{T,flag_trace}(traces)

@forward EpisodicTrajectory.traces Base.keys,
Base.haskey,
Base.getindex,
Base.push!,
Base.pop!,
Base.empty!

function isfull(t::EpisodicTrajectory{<:Any,F}) where {F}
    x = t.traces[F]
    (nframes(x) > 0) && select_last_frame(x)
end

#####
# CombinedTrajectory
#####

"""
    CombinedTrajectory(t1::AbstractTrajectory, t2::AbstractTrajectory)
"""
struct CombinedTrajectory{T1,T2} <: AbstractTrajectory
    t1::T1
    t2::T2
end

Base.haskey(t::CombinedTrajectory, s::Symbol) = haskey(t.t1, s) || haskey(t.t2, s)
Base.getindex(t::CombinedTrajectory, s::Symbol) =
    if haskey(t.t1, s)
        getindex(t.t1, s)
    elseif haskey(t.t2, s)
        getindex(t.t2, s)
    else
        throw(ArgumentError("unknown key: $s"))
    end

Base.keys(t::CombinedTrajectory) = (keys(t.t1)..., keys(t.t2)...)

Base.push!(t::CombinedTrajectory, kv::Pair{Symbol}) =
    if haskey(t.t1, first(kv))
        push!(t.t1, kv)
    elseif haskey(t.t2, first(kv))
        push!(t.t2, kv)
    else
        throw(ArgumentError("unknown kv: $kv"))
    end

Base.pop!(t::CombinedTrajectory, s::Symbol) =
    if haskey(t.t1, s)
        pop!(t.t1, s)
    elseif haskey(t.t2, s)
        pop!(t.t2, s)
    else
        throw(ArgumentError("unknown key: $s"))
    end

Base.pop!(t::CombinedTrajectory) = merge(pop!(t.t1), pop!(t.t2))

function Base.empty!(t::CombinedTrajectory)
    empty!(t.t1)
    empty!(t.t2)
end

isfull(t::CombinedTrajectory) = isfull(t.t1) && isfull(t.t2)

#####
# CircularCompactSATrajectory 
#####

const CircularCompactSATrajectory = CombinedTrajectory{
    <:SharedTrajectory{
        <:CircularArrayBuffer,
        <:NamedTuple{(:state, :next_state, :full_state)},
    },
    <:SharedTrajectory{
        <:CircularArrayBuffer,
        <:NamedTuple{(:action, :next_action, :full_action)},
    },
}

function CircularCompactSATrajectory(;
    capacity,
    state_type = Int,
    state_size = (),
    action_type = Int,
    action_size = (),
)
    CombinedTrajectory(
        SharedTrajectory(
            CircularArrayBuffer{state_type}(state_size..., capacity + 1),
            :state,
        ),
        SharedTrajectory(
            CircularArrayBuffer{action_type}(action_size..., capacity + 1),
            :action,
        ),
    )
end

#####
# CircularCompactSALTrajectory 
#####

const CircularCompactSALTrajectory = CombinedTrajectory{
    <:SharedTrajectory{
        <:CircularArrayBuffer,
        <:NamedTuple{
            (:legal_actions_mask, :next_legal_actions_mask, :full_legal_actions_mask),
        },
    },
    <:CircularCompactSATrajectory,
}

function CircularCompactSALTrajectory(;
    capacity,
    legal_actions_mask_size,
    legal_actions_mask_type = Bool,
    kw...,
)
    CombinedTrajectory(
        SharedTrajectory(
            CircularArrayBuffer{legal_actions_mask_type}(
                legal_actions_mask_size...,
                capacity + 1,
            ),
            :legal_actions_mask,
        ),
        CircularCompactSATrajectory(; capacity = capacity, kw...),
    )
end
#####
# CircularCompactSARTSATrajectory
#####

const CircularCompactSARTSATrajectory = CombinedTrajectory{
    <:Trajectory{
        <:NamedTuple{
            (:reward, :terminal),
            <:Tuple{<:CircularArrayBuffer,<:CircularArrayBuffer},
        },
    },
    <:CircularCompactSATrajectory,
}

function CircularCompactSARTSATrajectory(;
    capacity,
    reward_type = Float32,
    reward_size = (),
    terminal_type = Bool,
    terminal_size = (),
    kw...,
)
    CombinedTrajectory(
        Trajectory(
            reward = CircularArrayBuffer{reward_type}(reward_size..., capacity),
            terminal = CircularArrayBuffer{terminal_type}(terminal_size..., capacity),
        ),
        CircularCompactSATrajectory(; capacity = capacity, kw...),
    )
end

#####
# CircularCompactSALRTSALTrajectory
#####

const CircularCompactSALRTSALTrajectory = CombinedTrajectory{
    <:Trajectory{
        <:NamedTuple{
            (:reward, :terminal),
            <:Tuple{<:CircularArrayBuffer,<:CircularArrayBuffer},
        },
    },
    <:CircularCompactSALTrajectory,
}

function CircularCompactSALRTSALTrajectory(;
    capacity,
    reward_type = Float32,
    reward_size = (),
    terminal_type = Bool,
    terminal_size = (),
    kw...,
)
    CombinedTrajectory(
        Trajectory(
            reward = CircularArrayBuffer{reward_type}(reward_size..., capacity),
            terminal = CircularArrayBuffer{terminal_type}(terminal_size..., capacity),
        ),
        CircularCompactSALTrajectory(; capacity = capacity, kw...),
    )
end

#####
# CircularCompactPSARTSATrajectory
#####

const CircularCompactPSARTSATrajectory = CombinedTrajectory{
    <:Trajectory{
        <:NamedTuple{
            (:reward, :terminal, :priority),
            <:Tuple{<:CircularArrayBuffer,<:CircularArrayBuffer,<:SumTree},
        },
    },
    <:CircularCompactSATrajectory,
}

function CircularCompactPSARTSATrajectory(;
    capacity,
    priority_type = Float32,
    reward_type = Float32,
    reward_size = (),
    terminal_type = Bool,
    terminal_size = (),
    kw...,
)
    CombinedTrajectory(
        Trajectory(
            reward = CircularArrayBuffer{reward_type}(reward_size..., capacity),
            terminal = CircularArrayBuffer{terminal_type}(terminal_size..., capacity),
            priority = SumTree(priority_type, capacity),
        ),
        CircularCompactSATrajectory(; capacity = capacity, kw...),
    )
end

#####
# CircularCompactPSALRTSALTrajectory
#####

const CircularCompactPSALRTSALTrajectory = CombinedTrajectory{
    <:Trajectory{
        <:NamedTuple{
            (:reward, :terminal, :priority),
            <:Tuple{<:CircularArrayBuffer,<:CircularArrayBuffer,<:SumTree},
        },
    },
    <:CircularCompactSALTrajectory,
}

function CircularCompactPSALRTSALTrajectory(;
    capacity,
    priority_type = Float32,
    reward_type = Float32,
    reward_size = (),
    terminal_type = Bool,
    terminal_size = (),
    kw...,
)
    CombinedTrajectory(
        Trajectory(
            reward = CircularArrayBuffer{reward_type}(reward_size..., capacity),
            terminal = CircularArrayBuffer{terminal_type}(terminal_size..., capacity),
            priority = SumTree(priority_type, capacity),
        ),
        CircularCompactSALTrajectory(; capacity = capacity, kw...),
    )
end
