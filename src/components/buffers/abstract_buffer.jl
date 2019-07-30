export AbstractTurnBuffer, isfull, capacity, buffers,
       get_state, get_action, get_reward, get_terminal,
       push_state!, push_action!, push_reward!, push_terminal!

"""
    AbstractTurnBuffer{names, types} <: AbstractArray{NamedTuple{names, types}, 1}

`AbstractTurnBuffer` is supertype of a collection of buffers to store the interactions between agents and environments.
It is a subtype of `AbstractArray{NamedTuple{names, types}, 1}` where `names` specifies which fields are to store
and `types` is the coresponding types of the `names`.


| Required Methods| Brief Description |
|:----------------|:------------------|
| `Base.push!(b::AbstractTurnBuffer{names, types}, s[, a, r, d, s′, a′])` | Push a turn info into the buffer. According to different `names` and `types` of the buffer `b`, it may accept different number of arguments |
| `isfull(b)` | Check whether the buffer is full or not |
| `capacity(b)` | The maximum length of buffer |
| `Base.length(b)` | Return the length of buffer |
| `Base.getindex(b::AbstractTurnBuffer{names, types})` | Return a turn of type `NamedTuple{names, types}` |
| `Base.empty!(b)` | Reset the buffer |
| **Optional Methods** | |
| `Base.size(b)` | Return `(length(b),)` by default |
| `Base.isempty(b)` | Check whether the buffer is empty or not. Return `length(b) == 0` by default |
| `Base.lastindex(b)` | Return `length(b)` by default |
"""
abstract type AbstractTurnBuffer{names, types} <: AbstractArray{NamedTuple{names, types}, 1} end

function isfull end
function capacity end
buffers(b::AbstractTurnBuffer) = getfield(b, :buffers)

Base.size(b::AbstractTurnBuffer) = (length(b),)
Base.isempty(b::AbstractTurnBuffer) = length(b) == 0
Base.lastindex(b::AbstractTurnBuffer) = length(b)
Base.getindex(b::AbstractTurnBuffer, i::Int) = eltype(b)(x[i] for x in buffers(b))
Base.empty!(b::AbstractTurnBuffer) = for x in buffers(b) empty!(x) end

const SART = (:state, :action, :reward, :terminal)

get_state(b::AbstractTurnBuffer) = buffers(b).state
get_action(b::AbstractTurnBuffer) = buffers(b).action
get_reward(b::AbstractTurnBuffer) = buffers(b).reward
get_terminal(b::AbstractTurnBuffer) = buffers(b).terminal

push_state!(b::AbstractTurnBuffer, state) = push!(get_state(b), state)
push_action!(b::AbstractTurnBuffer, action) = push!(get_action(b), action)
push_reward!(b::AbstractTurnBuffer, reward) = push!(get_reward(b), reward)
push_terminal!(b::AbstractTurnBuffer, terminal) = push!(get_terminal(b), terminal)