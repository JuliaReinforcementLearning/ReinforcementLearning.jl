export Trajectory,
    PrioritizedTrajectory,
    DUMMY_TRAJECTORY,
    DummyTrajectory,
    CircularArrayTrajectory,
    CircularVectorTrajectory,
    CircularArraySARTTrajectory,
    CircularVectorSARTTrajectory,
    CircularVectorSARTSATrajectory,
    CircularArrayPSARTTrajectory,
    VectorTrajectory

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

const DUMMY_TRAJECTORY = Trajectory()
const DummyTrajectory = typeof(DUMMY_TRAJECTORY)

#####

function CircularArrayTrajectory(;capacity, kwargs...)
    Trajectory(map(kwargs.data) do x
        CircularArrayBuffer{eltype(first(x))}(last(x)..., capacity)
    end)
end

function CircularVectorTrajectory(;capacity, kwargs...)
    Trajectory(map(kwargs.data) do x
        CircularVectorBuffer{x}(capacity)
    end)
end

#####

const CircularArraySARTTrajectory = Trajectory{
    <:NamedTuple{
        (:state, :action, :reward, :terminal),
        <:Tuple{<:CircularArrayBuffer, <:CircularArrayBuffer, <:CircularArrayBuffer, <:CircularArrayBuffer}}}

CircularArraySARTTrajectory(;capacity::Int, state=Int=>(), action=Int=>(), reward=Float32=>(), terminal=Bool=>()) = merge(
    CircularArrayTrajectory(;capacity=capacity+1, state=state, action=action),
    CircularArrayTrajectory(;capacity=capacity, reward=reward, terminal=terminal),
)

const CircularArraySALRTTrajectory = Trajectory{
    <:NamedTuple{
        (:state, :action, :legal_actions_mask, :reward, :terminal),
        <:Tuple{<:CircularArrayBuffer, <:CircularArrayBuffer, <:CircularArrayBuffer, <:CircularArrayBuffer, <:CircularArrayBuffer}}}

CircularArraySALRTTrajectory(;capacity::Int, state=Int=>(), action=Int=>(), legal_actions_mask, reward=Float32=>(), terminal=Bool=>()) = merge(
    CircularArrayTrajectory(;capacity=capacity+1, state=state, action=action, legal_actions_mask=legal_actions_mask),
    CircularArrayTrajectory(;capacity=capacity, reward=reward, terminal=terminal),
)

#####

const CircularVectorSARTTrajectory = Trajectory{
    <:NamedTuple{
        (:state, :action, :reward, :terminal),
        <:Tuple{<:CircularVectorBuffer, <:CircularVectorBuffer, <:CircularVectorBuffer, <:CircularVectorBuffer}}}

CircularVectorSARTTrajectory(;capacity::Int, state=Int, action=Int, reward=Float32, terminal=Bool) = merge(
    CircularVectorTrajectory(;capacity=capacity+1, state=state, action=action),
    CircularVectorTrajectory(;capacity=capacity, reward=reward, terminal=terminal),
)

#####

const CircularVectorSARTSATrajectory = Trajectory{
    <:NamedTuple{
        (:state, :action, :reward, :terminal, :next_state, :next_action),
        <:Tuple{<:CircularVectorBuffer, <:CircularVectorBuffer, <:CircularVectorBuffer, <:CircularVectorBuffer, <:CircularVectorBuffer, <:CircularVectorBuffer}}}

CircularVectorSARTSATrajectory(;capacity::Int,  state=Int, action=Int, reward=Float32, terminal=Bool, next_state=state, next_action=action) = CircularVectorTrajectory(;capacity=capacity, state=state, action=action, reward=reward,terminal=terminal,next_state=next_state, next_action=next_action),

#####

function ElasticArrayTrajectory(;kwargs...)
    Trajectory(map(kwargs.data) do x
        ElasticArray{eltype(first(x))}(undef, last(x)..., 0)
    end)
end

#####
# VectorTrajectory
#####

function VectorTrajectory(;kwargs...)
    Trajectory(map(kwargs.data) do x
        Vector{x}()
    end)
end

#####

Base.@kwdef struct PrioritizedTrajectory{P,T} <: AbstractTrajectory
    priority::P
    traj::T
end

Base.keys(t::PrioritizedTrajectory) = (:priority, keys(t.traj)...)

Base.length(t::PrioritizedTrajectory) = length(t.priority)

Base.getindex(t::PrioritizedTrajectory, s::Symbol) = if s == :priority
    t.priority
else
    getindex(t.traj, s)
end

const CircularArrayPSARTTrajectory = PrioritizedTrajectory{<:SumTree, <:CircularArraySARTTrajectory}

CircularArrayPSARTTrajectory(;capacity, kwargs...) = PrioritizedTrajectory(
    SumTree(capacity),
    CircularArraySARTTrajectory(;capacity=capacity, kwargs...)
)

#####
# Common
#####

function Base.length(t::Union{<:CircularArraySARTTrajectory,<:CircularVectorSARTSATrajectory})
    x = t[:terminal]
    size(x, ndims(x))
end
