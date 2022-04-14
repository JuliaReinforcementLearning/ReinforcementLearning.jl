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
import DataStructures.Set

#####
# Trajectory
#####

"""
    Trajectory(;[trace_name=trace_container]..., is_episodic = true)

Mainly a simple wrapper of `NamedTuple`. 
Set `is_episodic = false` when working with non-episodic environments (i.e. infinite horizon) that stop after a given number of steps to avoid multiplying the value if the last state by 0 when bootstrapping TD targets
"""
struct Trajectory{T} <: AbstractTrajectory
    traces::T
    is_episodic::Bool
    last_states_idxs::Set{Int}
end

Trajectory(; is_episodic = true, kwargs...) = Trajectory(values(kwargs), is_episodic, Set{Int64}())

@forward Trajectory.traces Base.getindex, Base.keys

Base.merge(a::Trajectory, b::Trajectory) = Trajectory(merge(a.traces, b.traces), a.is_episodic, Set{Int64}())
Base.merge(a::Trajectory, b::NamedTuple) = Trajectory(merge(a.traces, b), a.is_episodic, Set{Int64}())
Base.merge(a::NamedTuple, b::Trajectory) = Trajectory(merge(a, b.traces), b.is_episodic, Set{Int64}())

#####

"""
    CircularArrayTrajectory(; capacity::Int, kw::Pair{<:DataType, <:Tuple{Vararg{Int}}}...)

A specialized [`Trajectory`](@ref) which uses
[`CircularArrayBuffer`](https://github.com/JuliaReinforcementLearning/CircularArrayBuffers.jl#usage)
as the underlying storage. `kw` specifies the name, the element type and the
size of each trace. `capacity` is used to define the maximum length of the
underlying buffer.

See also [`CircularArraySARTTrajectory`](@ref),
[`CircularArraySLARTTrajectory`](@ref), [`CircularArrayPSARTTrajectory`](@ref).
"""
function CircularArrayTrajectory(; capacity, is_episodic = true, kwargs...)
    Trajectory(map(values(kwargs)) do x
        CircularArrayBuffer{eltype(first(x))}(last(x)..., capacity)
    end, 
    is_episodic, Set{Int64}())
end

"""
    CircularVectorTrajectory(;capacity, kw::DataType)

Similar to [`CircularArrayTrajectory`](@ref), except that the underlying storage is
[`CircularVectorBuffer`](https://github.com/JuliaReinforcementLearning/CircularArrayBuffers.jl#usage).

!!! note
    Note the different type of the `kw` between `CircularVectorTrajectory` and `CircularArrayTrajectory`. With 
    [`CircularVectorBuffer`](https://github.com/JuliaReinforcementLearning/CircularArrayBuffers.jl#usage)
    as the underlying storage, we don't need the size info.

See also [`CircularVectorSARTTrajectory`](@ref), [`CircularVectorSARTSATrajectory`](@ref).
"""
function CircularVectorTrajectory(; capacity, is_episodic = true, kwargs...)
    Trajectory(map(values(kwargs)) do x
        CircularVectorBuffer{x}(capacity)
    end,
    is_episodic, Set{Int64}())
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

"""
    CircularArraySARTTrajectory(;capacity::Int, kw...)

A specialized [`CircularArrayTrajectory`](@ref) with traces of [`SART`](@ref).
Note that the capacity of the `:state` and `:action` trace is one step longer
than the capacity of the `:reward` and `:terminal` trace, so that we can reuse
the same trace to represent the next state and next action in a typical
transition in reinforcement learning.

# Keyword arguments

- `capacity::Int`, the maximum number of transitions.
- `state::Pair{<:DataType, <:Tuple{Vararg{Int}}}` = `Int => ()`,
- `action::Pair{<:DataType, <:Tuple{Vararg{Int}}}` = `Int => ()`,
- `reward::Pair{<:DataType, <:Tuple{Vararg{Int}}}` = `Float32 => ()`,
- `terminal::Pair{<:DataType, <:Tuple{Vararg{Int}}}` = `Bool => ()`,

# Example

```julia-repl
julia> t = CircularArraySARTTrajectory(;
           capacity = 3,
           state = Vector{Int} => (4,),
           action = Int => (),
           reward = Float32 => (),
           terminal = Bool => (),
       )
Trajectory of 4 traces:
:state 4×0 CircularArrayBuffers.CircularArrayBuffer{Int64, 2}
:action 0-element CircularArrayBuffers.CircularVectorBuffer{Int64}
:reward 0-element CircularArrayBuffers.CircularVectorBuffer{Float32}
:terminal 0-element CircularArrayBuffers.CircularVectorBuffer{Bool}


julia> for i in 1:4
           push!(t;state=ones(Int, 4) .* i, action = i, reward=i/2, terminal=iseven(i))
       end

julia> push!(t;state=ones(Int,4) .* 5, action = 5)

julia> t[:state]
4×4 CircularArrayBuffers.CircularArrayBuffer{Int64, 2}:
 2  3  4  5
 2  3  4  5
 2  3  4  5
 2  3  4  5

julia> t[:action]
4-element CircularArrayBuffers.CircularVectorBuffer{Int64}:
 2
 3
 4
 5

julia> t[:reward]
3-element CircularArrayBuffers.CircularVectorBuffer{Float32}:
 1.0
 1.5
 2.0

julia> t[:terminal]
3-element CircularArrayBuffers.CircularVectorBuffer{Bool}:
 1
 0
 1
```
"""
CircularArraySARTTrajectory(;
    capacity::Int,
    state = Int => (),
    action = Int => (),
    reward = Float32 => (),
    terminal = Bool => (),
    is_episodic = true) = merge(
    CircularArrayTrajectory(; capacity = capacity , state = state, action = action, is_episodic = is_episodic),
    CircularArrayTrajectory(; capacity = capacity, reward = reward, terminal = terminal, is_episodic = is_episodic),
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

"Similar to [`CircularArraySARTTrajectory`](@ref) with an extra `legal_actions_mask` trace."
CircularArraySLARTTrajectory(;
    capacity::Int,
    state = Int => (),
    legal_actions_mask,
    action = Int => (),
    reward = Float32 => (),
    terminal = Bool => (),
    is_episodic = true
) = merge(
    CircularArrayTrajectory(;
        capacity = capacity ,
        state = state,
        legal_actions_mask = legal_actions_mask,
        action = action,
        is_episodic = is_episodic
    ),
    CircularArrayTrajectory(; capacity = capacity, reward = reward, terminal = terminal, is_episodic = is_episodic),
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

"""
    CircularVectorSARTTrajectory(;capacity, kw::DataType...)

A specialized [`CircularVectorTrajectory`](@ref) with traces of [`SART`](@ref).
Note that the capacity of traces `:state` and `:action` are one step longer than
the traces of `:reward` and `:terminal`, so that we can reuse the same
underlying storage to represent the next state and next action in a typical
transition in reinforcement learning.

# Keyword arguments

- `capacity::Int`
- `state` = `Int`,
- `action` = `Int`,
- `reward` = `Float32`,
- `terminal` = `Bool`,

# Example

```julia-repl
julia> t = CircularVectorSARTTrajectory(;
           capacity = 3,
           state = Vector{Int},
           action = Int,
           reward = Float32,
           terminal = Bool,
       )
Trajectory of 4 traces:
:state 0-element CircularArrayBuffers.CircularVectorBuffer{Vector{Int64}}
:action 0-element CircularArrayBuffers.CircularVectorBuffer{Int64}
:reward 0-element CircularArrayBuffers.CircularVectorBuffer{Float32}
:terminal 0-element CircularArrayBuffers.CircularVectorBuffer{Bool}


julia> for i in 1:4
           push!(t;state=ones(Int, 4) .* i, action = i, reward=i/2, terminal=iseven(i))
       end

julia> push!(t;state=ones(Int,4) .* 5, action = 5)

julia> t[:state]
4-element CircularArrayBuffers.CircularVectorBuffer{Vector{Int64}}:
 [2, 2, 2, 2]
 [3, 3, 3, 3]
 [4, 4, 4, 4]
 [5, 5, 5, 5]

julia> t[:action]
4-element CircularArrayBuffers.CircularVectorBuffer{Int64}:
 2
 3
 4
 5

julia> t[:reward]
3-element CircularArrayBuffers.CircularVectorBuffer{Float32}:
 1.0
 1.5
 2.0

julia> t[:terminal]
3-element CircularArrayBuffers.CircularVectorBuffer{Bool}:
 1
 0
 1
```
"""
CircularVectorSARTTrajectory(;
    capacity::Int,
    state = Int,
    action = Int,
    reward = Float32,
    terminal = Bool,
    is_episodic = true
) = merge(
    CircularVectorTrajectory(; capacity = capacity , state = state, action = action, is_episodic = is_episodic),
    CircularVectorTrajectory(; capacity = capacity, reward = reward, terminal = terminal, is_episodic = is_episodic),
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

"Similar to [`CircularVectorSARTTrajectory`](@ref) with another two traces of `(:next_state, :next_action)`"
CircularVectorSARTSATrajectory(;
    capacity::Int,
    state = Int,
    action = Int,
    reward = Float32,
    terminal = Bool,
    next_state = state,
    next_action = action,
    is_episodic = true
) = CircularVectorTrajectory(;
    capacity = capacity,
    state = state,
    action = action,
    reward = reward,
    terminal = terminal,
    next_state = next_state,
    next_action = next_action,
    is_episodic = is_episodic
)

#####

"""
    ElasticArrayTrajectory(;[trace_name::Pair{<:DataType, <:Tuple{Vararg{Int}}}]...)

A specialized [`Trajectory`](@ref) which uses [`ElasticArray`](https://github.com/JuliaArrays/ElasticArrays.jl) as the underlying
storage. See also [`ElasticSARTTrajectory`](@ref).
"""
function ElasticArrayTrajectory(; is_episodic = true, kwargs...)
    Trajectory(map(values(kwargs)) do x
        ElasticArray{eltype(first(x))}(undef, last(x)..., 0)
    end,
    is_episodic, Set{Int64}())
end

const ElasticSARTTrajectory = Trajectory{
    <:NamedTuple{SART,<:Tuple{<:ElasticArray,<:ElasticArray,<:ElasticArray,<:ElasticArray}},
}

"""
    ElasticSARTTrajectory(;kw...)

A specialized [`ElasticArrayTrajectory`](@ref) with traces of [`SART`](@ref).

# Keyword arguments

- `state::Pair{<:DataType, <:Tuple{Vararg{Int}}}` = `Int => ()`, by default it
  means the state is a scalar of `Int`.
- `action::Pair{<:DataType, <:Tuple{Vararg{Int}}}` = `Int => ()`,
- `reward::Pair{<:DataType, <:Tuple{Vararg{Int}}}` = `Float32 => ()`,
- `terminal::Pair{<:DataType, <:Tuple{Vararg{Int}}}` = `Bool => ()`,

# Example

```julia-repl
julia> t = ElasticSARTTrajectory(;
           state = Vector{Int} => (4,),
           action = Int => (),
           reward = Float32 => (),
           terminal = Bool => (),
       )
Trajectory of 4 traces:
:state 4×0 ElasticArrays.ElasticMatrix{Int64, Vector{Int64}}
:action 0-element ElasticArrays.ElasticVector{Int64, Vector{Int64}}
:reward 0-element ElasticArrays.ElasticVector{Float32, Vector{Float32}}
:terminal 0-element ElasticArrays.ElasticVector{Bool, Vector{Bool}}


julia> for i in 1:4
           push!(t;state=ones(Int, 4) .* i, action = i, reward=i/2, terminal=iseven(i))
       end

julia> push!(t;state=ones(Int,4) .* 5, action = 5)

julia> t
Trajectory of 4 traces:
:state 4×5 ElasticArrays.ElasticMatrix{Int64, Vector{Int64}}
:action 5-element ElasticArrays.ElasticVector{Int64, Vector{Int64}}
:reward 4-element ElasticArrays.ElasticVector{Float32, Vector{Float32}}
:terminal 4-element ElasticArrays.ElasticVector{Bool, Vector{Bool}}

julia> t[:state]
4×5 ElasticArrays.ElasticMatrix{Int64, Vector{Int64}}:
 1  2  3  4  5
 1  2  3  4  5
 1  2  3  4  5
 1  2  3  4  5

julia> t[:action]
5-element ElasticArrays.ElasticVector{Int64, Vector{Int64}}:
 1
 2
 3
 4
 5

julia> t[:reward]
4-element ElasticArrays.ElasticVector{Float32, Vector{Float32}}:
 0.5
 1.0
 1.5
 2.0

julia> t[:terminal]
4-element ElasticArrays.ElasticVector{Bool, Vector{Bool}}:
 0
 1
 0
 1

julia> empty!(t)

julia> t
Trajectory of 4 traces:
:state 4×0 ElasticArrays.ElasticMatrix{Int64, Vector{Int64}}
:action 0-element ElasticArrays.ElasticVector{Int64, Vector{Int64}}
:reward 0-element ElasticArrays.ElasticVector{Float32, Vector{Float32}}
:terminal 0-element ElasticArrays.ElasticVector{Bool, Vector{Bool}}
```

"""
function ElasticSARTTrajectory(;
    state = Int => (),
    action = Int => (),
    reward = Float32 => (),
    terminal = Bool => (),
    is_episodic = true
)
    ElasticArrayTrajectory(;
        state = state,
        action = action,
        reward = reward,
        terminal = terminal,
        is_episodic = is_episodic
    )
end

#####
# VectorTrajectory
#####

"""
    VectorTrajectory(;[trace_name::DataType]...)

A [`Trajectory`](@ref) with each trace using a `Vector` as the storage.
"""
function VectorTrajectory(; is_episodic = true, kwargs...)
    Trajectory(map(values(kwargs)) do x
        Vector{x}()
    end,
    is_episodic, Set{Int64}())
end

const VectorSARTTrajectory =
    Trajectory{<:NamedTuple{SART,<:Tuple{<:Vector,<:Vector,<:Vector,<:Vector}}}

"""
    VectorSARTTrajectory(;kw...)

A specialized [`VectorTrajectory`] with traces of [`SART`](@ref).

# Keyword arguments

- `state::DataType = Int`
- `action::DataType = Int`
- `reward::DataType = Float32`
- `terminal::DataType = Bool`
"""
function VectorSARTTrajectory(;
    state = Int,
    action = Int,
    reward = Float32,
    terminal = Bool,
    is_episodic = true
)
    VectorTrajectory(; state = state, action = action, reward = reward, terminal = terminal, is_episodic = is_episodic)
end

const VectorSATrajectory =
    Trajectory{<:NamedTuple{(:state, :action),<:Tuple{<:Vector,<:Vector}}}

"""
    VectorSATrajectory(;kw...)

A specialized [`VectorTrajectory`] with traces of `(:state, :action)`.

# Keyword arguments

- `state::DataType = Int`
- `action::DataType = Int`
"""
function VectorSATrajectory(; state = Int, action = Int, is_episodic = true)
    VectorTrajectory(; state = state, action = action, is_episodic = is_episodic)
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

CircularArrayPSARTTrajectory(; capacity, is_episodic = true, kwargs...) = PrioritizedTrajectory(
    CircularArraySARTTrajectory(; capacity = capacity, is_episodic = is_episodic, kwargs...),
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
