export CircularTurnBuffer, circular_RTSA_buffer, capacity, isfull, circular_PRTSA_buffer

import StatsBase: sample
using .Utils: SumTree

"""
    CircularTurnBuffer{names,types,Tbs}

Contain a collection of buffers (mainly are [`CircularArrayBuffer`](@ref), but not restriced to it) to represent the interractions between agents and environments. This struct itself is very simple, some commonly used buffers are provided by:

- [`circular_PRTSA_buffer`](@ref)
- [`circular_PRTSA_buffer`](@ref)

# Fields

- `buffers`: a tuple of inner buffers

"""
struct CircularTurnBuffer{names,types,Tbs} <: AbstractTurnBuffer{names,types}
    buffers::Tbs
end

"""
    sample(b::CircularTurnBuffer, batch_size, n_step, stack_size)

!!! note
    `stack_size` can be an `Int`, and then the size of state is expanded in the last dimension.
    For example if the original state is an image of size `(84, 84)`, then the size of new state is `(84, 84, stack_size)`.
"""
function sample(b::CircularTurnBuffer, batch_size, n_step, stack_size)
    inds = sample_indices(b, batch_size, n_step, stack_size)
    inds, consecutive_view(b, inds, n_step, stack_size)
end

sample_indices(b::CircularTurnBuffer, batch_size::Int, n_step::Int, stack_size::Nothing) = sample_indices(b, batch_size, n_step, 1)

#####
# RTSA
#####

"""
    circular_RTSA_buffer(;kwargs...) -> CircularTurnBuffer

A helper function to help generate a [`CircularTurnBuffer`](@ref) with **RTSA** (**R**eward, **T**erminal, **S**tate, **A**ction) fields as buffers.
    
# Keywords

## Necessary

- `capacity::Int`: the maximum length of the buffer

## Optional

- `state_eltype::Type=Int`: the type of the state field in an [`Observation`](@ref), `Int` by default
- `state_size::NTuple{N, Int}=()`: the size of the state field in an [`Observation`](@ref), the `N` must match `ndims(state_eltype)`. Since the default `state_eltype` is `Int`, it is an empty tuple here by default.
- `action_eltype::Type=Int`: similar to `state_eltype`
- `action_size::NTuple{N, Int}=()`: similar to `state_size`
- `reward_eltype::Type=Float32`: similar to `state_eltype`
- `reward_size::NTuple{N, Int}=()`: similar to `state_size`
- `terminal_eltype::Type=Bool`: similar to `state_eltype`
- `terminal_size::NTuple{N, Int}=()`: similar to `state_size`
    
The following picture will help you understand how the data are organized.

![](../assets/img/circular_RTSA_buffer.png)

# Examples

```@julia-repl
julia> using ReinforcementLearning

julia> b = circular_RTSA_buffer(;capacity=2, state_eltype=Array{Float32, 2}, state_size=(2, 2))
0-element CircularTurnBuffer{(:reward, :terminal, :state, :action),Tuple{Float32,Bool,Array{Float32,2},Int64},NamedTuple{(:reward, :terminal, :state, :action),Tuple{CircularArrayBuffer{Float32,Float32,1},CircularArrayBuffer{Bool,Bool,1},CircularArrayBuffer{Array{Float32,2},Float32,3},CircularArrayBuffer{Int64,Int64,1}}}}

julia> push!(b; reward = 0.0, terminal = true, state = Float32[0 0; 0 0], action = 0)

julia> length(b)
0

julia> b.buffers.reward
1-element CircularArrayBuffer{Float32,Float32,1}:
 0.0

julia> b.buffers.terminal
1-element CircularArrayBuffer{Bool,Bool,1}:
 1

julia> b.buffers.state
2×2×1 CircularArrayBuffer{Array{Float32,2},Float32,3}:
[:, :, 1] =
 0.0  0.0
 0.0  0.0

julia> b.buffers.action
1-element CircularArrayBuffer{Int64,Int64,1}:
 0

julia> push!(b; reward = 1.0, terminal = false, state = Float32[1 1; 1 1], action = 1)

julia> b
1-element CircularTurnBuffer{(:reward, :terminal, :state, :action),Tuple{Float32,Bool,Array{Float32,2},Int64},NamedTuple{(:reward, :terminal, :state, :action),Tuple{CircularArrayBuffer{Float32,Float32,1},CircularArrayBuffer{Bool,Bool,1},CircularArrayBuffer{Array{Float32,2},Float32,3},CircularArrayBuffer{Int64,Int64,1}}}}:
 (state = Float32[0.0 0.0; 0.0 0.0], action = 0, reward = 1.0f0, terminal = false, next_state = Float32[1.0 1.0; 1.0 1.0], next_action = 1)

julia> push!(b; reward = 2.0, terminal = true, state = Float32[2 2; 2 2], action = 2)

julia> b
2-element CircularTurnBuffer{(:reward, :terminal, :state, :action),Tuple{Float32,Bool,Array{Float32,2},Int64},NamedTuple{(:reward, :terminal, :state, :action),Tuple{CircularArrayBuffer{Float32,Float32,1},CircularArrayBuffer{Bool,Bool,1},CircularArrayBuffer{Array{Float32,2},Float32,3},CircularArrayBuffer{Int64,Int64,1}}}}:
 (state = Float32[0.0 0.0; 0.0 0.0], action = 0, reward = 1.0f0, terminal = false, next_state = Float32[1.0 1.0; 1.0 1.0], next_action = 1)
 (state = Float32[1.0 1.0; 1.0 1.0], action = 1, reward = 2.0f0, terminal = true, next_state = Float32[2.0 2.0; 2.0 2.0], next_action = 2) 

julia> push!(b; reward = 3.0, terminal = false, state = Float32[3 3; 3 3], action = 3)

julia> b
2-element CircularTurnBuffer{(:reward, :terminal, :state, :action),Tuple{Float32,Bool,Array{Float32,2},Int64},NamedTuple{(:reward, :terminal, :state, :action),Tuple{CircularArrayBuffer{Float32,Float32,1},CircularArrayBuffer{Bool,Bool,1},CircularArrayBuffer{Array{Float32,2},Float32,3},CircularArrayBuffer{Int64,Int64,1}}}}:
 (state = Float32[1.0 1.0; 1.0 1.0], action = 1, reward = 2.0f0, terminal = true, next_state = Float32[2.0 2.0; 2.0 2.0], next_action = 2) 
 (state = Float32[2.0 2.0; 2.0 2.0], action = 2, reward = 3.0f0, terminal = false, next_state = Float32[3.0 3.0; 3.0 3.0], next_action = 3)

julia> b.buffers.state
2×2×3 CircularArrayBuffer{Array{Float32,2},Float32,3}:
[:, :, 1] =
 1.0  1.0
 1.0  1.0

[:, :, 2] =
 2.0  2.0
 2.0  2.0

[:, :, 3] =
 3.0  3.0
 3.0  3.0

julia> b.buffers.reward
3-element CircularArrayBuffer{Float32,Float32,1}:
 1.0
 2.0
 3.0

julia> length(b)
2
```
"""
function circular_RTSA_buffer(
    ;
    capacity,
    state_eltype = Int,
    state_size = (),
    action_eltype = Int,
    action_size = (),
    reward_eltype = Float32,
    reward_size = (),
    terminal_eltype = Bool,
    terminal_size = (),
)
    capacity += 1  # we need to store extra dummy (reward, terminal)
    buffers = (
        reward = CircularArrayBuffer{reward_eltype}(reward_size..., capacity),
        terminal = CircularArrayBuffer{terminal_eltype}(terminal_size..., capacity),
        state = CircularArrayBuffer{state_eltype}(state_size..., capacity),
        action = CircularArrayBuffer{action_eltype}(action_size..., capacity),
    )
    CircularTurnBuffer{
        RTSA,
        Tuple{reward_eltype,terminal_eltype,state_eltype,action_eltype},
        typeof(buffers),
    }(buffers)
end

sample_indices(b::CircularTurnBuffer{RTSA}, batch_size::Int, n_step::Int, stack_size::Int) =
    rand(stack_size:length(b)-n_step, batch_size)

#####
# PRTSA
#####

"""
    circular_PRTSA_buffer(;kwargs...) -> CircularTurnBuffer

The only difference compared to [`circular_RTSA_buffer`](@ref) is that a new field named `priority` is added. Notice that the struct of `priority` is not a [`CircularArrayBuffer`](@ref) but a [`SumTree`](@ref).

# Keywords

## Necessary

- `capacity::Int`: the maximum length of the buffer

## Optional

- `state_eltype::Type=Int`: the type of the state field in an [`Observation`](@ref), `Int` by default
- `state_size::NTuple{N, Int}=()`: the size of the state field in an [`Observation`](@ref), the `N` must match `ndims(state_eltype)`. Since the default `state_eltype` is `Int`, it is an empty tuple here by default.
- `action_eltype::Type=Int`: similar to `state_eltype`
- `action_size::NTuple{N, Int}=()`: similar to `state_size`
- `reward_eltype::Type=Float32`: similar to `state_eltype`
- `reward_size::NTuple{N, Int}=()`: similar to `state_size`
- `terminal_eltype::Type=Bool`: similar to `state_eltype`
- `terminal_size::NTuple{N, Int}=()`: similar to `state_size`
- `priority_eltype::Type=Float64`: the type of the `priority` field

"""
function circular_PRTSA_buffer(
    ;
    capacity,
    state_eltype = Int,
    state_size = (),
    action_eltype = Int,
    action_size = (),
    reward_eltype = Float64,
    reward_size = (),
    terminal_eltype = Bool,
    terminal_size = (),
    priority_eltype = Float64,
)
    capacity += 1  # we need to store extra dummy (reward, terminal)
    buffers = (
        priority = SumTree(priority_eltype, capacity),
        reward = CircularArrayBuffer{reward_eltype}(reward_size..., capacity),
        terminal = CircularArrayBuffer{terminal_eltype}(terminal_size..., capacity),
        state = CircularArrayBuffer{state_eltype}(state_size..., capacity),
        action = CircularArrayBuffer{action_eltype}(action_size..., capacity),
    )
    CircularTurnBuffer{
        PRTSA,
        Tuple{priority_eltype,reward_eltype,terminal_eltype,state_eltype,action_eltype},
        typeof(buffers),
    }(buffers)
end

function sample_indices(b::CircularTurnBuffer{PRTSA}, batch_size::Int, n_step::Int, stack_size::Int)
    inds = Vector{Int}(undef, batch_size)
    for i = 1:length(inds)
        ind, p = sample(priority(b))
        while ind <= stack_size || ind > length(b) - n_step + 1
            ind, p = sample(priority(b))
        end
        inds[i] = ind - 1  # !!! left shift by 1 because we are padding for priority
    end
    inds
end