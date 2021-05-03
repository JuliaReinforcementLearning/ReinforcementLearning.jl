export Trajectory,
    PrioritizedTrajectory,
    CircularArrayTrajectory,
    CircularVectorTrajectory,
    CircularArraySARTTrajectory,
    CircularArraySLARTTrajectory,
    CircularVectorSARTTrajectory,
    CircularVectorSARTSATrajectory,
    CircularArrayPSARTTrajectory,
    ElasticArrayTrajectory,
    ElasticSARTTrajectory,
    VectorTrajectory,
    VectorSATrajectory,
    VectorSARTTrajectory

using MacroTools: @forward
using ElasticArrays
using CircularArrayBuffers: CircularArrayBuffer, CircularVectorBuffer

#####
# Trajectory
#####

"""
    Trajectory(;[trace_name=trace_container]...)

A simple wrapper of `NamedTuple`.
Define our own type here to avoid type piracy with `NamedTuple`
"""
struct Trajectory{T} <: AbstractTrajectory
    traces::T
end

Trajectory(; kwargs...) = Trajectory(kwargs.data)

@forward Trajectory.traces Base.getindex, Base.keys

Base.merge(a::Trajectory, b::Trajectory) = Trajectory(merge(a.traces, b.traces))
Base.merge(a::Trajectory, b::NamedTuple) = Trajectory(merge(a.traces, b))
Base.merge(a::NamedTuple, b::Trajectory) = Trajectory(merge(a, b.traces))

#####

function CircularArrayTrajectory(; capacity, kwargs...)
    Trajectory(map(kwargs.data) do x
        CircularArrayBuffer{eltype(first(x))}(last(x)..., capacity)
    end)
end

function CircularVectorTrajectory(; capacity, kwargs...)
    Trajectory(map(kwargs.data) do x
        CircularVectorBuffer{x}(capacity)
    end)
end

#####

const CircularArraySARTTrajectory = Trajectory{
    <:NamedTuple{
        SART,
        <:Tuple{
            <:CircularArrayBuffer,
            <:CircularArrayBuffer,
            <:CircularArrayBuffer,
            <:CircularArrayBuffer,
        },
    },
}

CircularArraySARTTrajectory(;
    capacity::Int,
    state = Int => (),
    action = Int => (),
    reward = Float32 => (),
    terminal = Bool => (),
) = merge(
    CircularArrayTrajectory(; capacity = capacity + 1, state = state, action = action),
    CircularArrayTrajectory(; capacity = capacity, reward = reward, terminal = terminal),
)

const CircularArraySLARTTrajectory = Trajectory{
    <:NamedTuple{
        SLART,
        <:Tuple{
            <:CircularArrayBuffer,
            <:CircularArrayBuffer,
            <:CircularArrayBuffer,
            <:CircularArrayBuffer,
            <:CircularArrayBuffer,
        },
    },
}

CircularArraySLARTTrajectory(;
    capacity::Int,
    state = Int => (),
    legal_actions_mask,
    action = Int => (),
    reward = Float32 => (),
    terminal = Bool => (),
) = merge(
    CircularArrayTrajectory(;
        capacity = capacity + 1,
        state = state,
        legal_actions_mask = legal_actions_mask,
        action = action,
    ),
    CircularArrayTrajectory(; capacity = capacity, reward = reward, terminal = terminal),
)

#####

const CircularVectorSARTTrajectory = Trajectory{
    <:NamedTuple{
        SART,
        <:Tuple{
            <:CircularVectorBuffer,
            <:CircularVectorBuffer,
            <:CircularVectorBuffer,
            <:CircularVectorBuffer,
        },
    },
}

CircularVectorSARTTrajectory(;
    capacity::Int,
    state = Int,
    action = Int,
    reward = Float32,
    terminal = Bool,
) = merge(
    CircularVectorTrajectory(; capacity = capacity + 1, state = state, action = action),
    CircularVectorTrajectory(; capacity = capacity, reward = reward, terminal = terminal),
)

#####

const CircularVectorSARTSATrajectory = Trajectory{
    <:NamedTuple{
        SARTSA,
        <:Tuple{
            <:CircularVectorBuffer,
            <:CircularVectorBuffer,
            <:CircularVectorBuffer,
            <:CircularVectorBuffer,
            <:CircularVectorBuffer,
            <:CircularVectorBuffer,
        },
    },
}

CircularVectorSARTSATrajectory(;
    capacity::Int,
    state = Int,
    action = Int,
    reward = Float32,
    terminal = Bool,
    next_state = state,
    next_action = action,
) = CircularVectorTrajectory(;
    capacity = capacity,
    state = state,
    action = action,
    reward = reward,
    terminal = terminal,
    next_state = next_state,
    next_action = next_action,
)

#####

function ElasticArrayTrajectory(; kwargs...)
    Trajectory(map(kwargs.data) do x
        ElasticArray{eltype(first(x))}(undef, last(x)..., 0)
    end)
end

const ElasticSARTTrajectory = Trajectory{
    <:NamedTuple{SART,<:Tuple{<:ElasticArray,<:ElasticArray,<:ElasticArray,<:ElasticArray}},
}

function ElasticSARTTrajectory(;
    state = Int => (),
    action = Int => (),
    reward = Float32 => (),
    terminal = Bool => (),
)
    ElasticArrayTrajectory(;
        state = state,
        action = action,
        reward = reward,
        terminal = terminal,
    )
end

#####
# VectorTrajectory
#####

function VectorTrajectory(; kwargs...)
    Trajectory(map(kwargs.data) do x
        Vector{x}()
    end)
end

const VectorSARTTrajectory =
    Trajectory{<:NamedTuple{SART,<:Tuple{<:Vector,<:Vector,<:Vector,<:Vector}}}

function VectorSARTTrajectory(;
    state = Int,
    action = Int,
    reward = Float32,
    terminal = Bool,
)
    VectorTrajectory(; state = state, action = action, reward = reward, terminal = terminal)
end

const VectorSATrajectory =
    Trajectory{<:NamedTuple{(:state, :action),<:Tuple{<:Vector,<:Vector}}}

function VectorSATrajectory(; state = Int, action = Int)
    VectorTrajectory(; state = state, action = action)
end
#####

Base.@kwdef struct PrioritizedTrajectory{T,P} <: AbstractTrajectory
    traj::T
    priority::P
end

Base.keys(t::PrioritizedTrajectory) = (:priority, keys(t.traj)...)

Base.length(t::PrioritizedTrajectory) = length(t.priority)

Base.getindex(t::PrioritizedTrajectory, s::Symbol) =
    if s == :priority
        t.priority
    else
        getindex(t.traj, s)
    end

const CircularArrayPSARTTrajectory =
    PrioritizedTrajectory{<:SumTree,<:CircularArraySARTTrajectory}

CircularArrayPSARTTrajectory(; capacity, kwargs...) = PrioritizedTrajectory(
    CircularArraySARTTrajectory(; capacity = capacity, kwargs...),
    SumTree(capacity),
)

#####
# Common
#####

function Base.length(
    t::Union{
        CircularArraySARTTrajectory,
        CircularArraySLARTTrajectory,
        CircularVectorSARTSATrajectory,
        ElasticSARTTrajectory,
    },
)
    x = t[:terminal]
    size(x, ndims(x))
end

Base.length(t::VectorSARTTrajectory) = length(t[:terminal])
Base.length(t::VectorSATrajectory) = length(t[:action])
