export AbstractTurnBuffer, buffers, RTSA, isfull, consecutive_view

"""
    AbstractTurnBuffer{names, types} <: AbstractArray{NamedTuple{names, types}, 1}

`AbstractTurnBuffer` is supertype of a collection of buffers to store the interactions between agents and environments.
It is a subtype of `AbstractArray{NamedTuple{names, types}, 1}` where `names` specifies which fields are to store
and `types` is the coresponding types of the `names`.


| Required Methods| Brief Description |
|:----------------RTSA----------------|
| `Base.push!(b::AbstractTurnBuffer{names, types}, s[, a, r, d, s′, a′])` | Push a turn info into the buffer. According to different `names` and `types` of the buffer `b`, it may accept different number of arguments |
| `isfull(b)` | Check whether the buffer is full or not |
| `Base.length(b)` | Return the length of buffer |
| `Base.getindex(b::AbstractTurnBuffer{names, types})` | Return a turn of type `NamedTuple{names, types}` |
| `Base.empty!(b)` | Reset the buffer |
| **Optional Methods** | |
| `Base.size(b)` | Return `(length(b),)` by default |
| `Base.isempty(b)` | Check whether the buffer is empty or not. Return `length(b) == 0` by default |
| `Base.lastindex(b)` | Return `length(b)` by default |
"""
abstract type AbstractTurnBuffer{names, types} <: AbstractArray{NamedTuple{names, types}, 1} end

buffers(b::AbstractTurnBuffer) = getfield(b, :buffers)

Base.size(b::AbstractTurnBuffer) = (length(b),)
Base.lastindex(b::AbstractTurnBuffer) = length(b)
Base.isempty(b::AbstractTurnBuffer) = all(isempty(x) for x in buffers(b))
Base.empty!(b::AbstractTurnBuffer) = for x in buffers(b) empty!(x) end
Base.getindex(b::AbstractTurnBuffer{names, types}, i::Int) where {names, types} = NamedTuple{names, types}(Tuple(x[i] for x in buffers(b)))
Base.length(b::AbstractTurnBuffer) = minimum(length(x) for x in buffers(b))
isfull(b::AbstractTurnBuffer) = all(isfull(x) for x in buffers(b))

function Base.push!(b::AbstractTurnBuffer, args...)
    for (b, x) in zip(buffers(b), args)
        push!(b, x)
    end
end

function Base.push!(b::AbstractTurnBuffer; kw...)
    for (k, v) in kw
        push!(getproperty(buffers(b), k), v)
    end
end

function Base.push!(b::AbstractTurnBuffer, experience::Pair{<:EnvObservation})
    obs, a = experience
    push!(b;
    state=state(obs),
    reward =reward(obs),
    terminal=terminal(obs),
    action=a,
    obs.meta...)
end

#####
# RTSA
#####

state(b::AbstractTurnBuffer) = buffers(b).state
action(b::AbstractTurnBuffer) = buffers(b).action
reward(b::AbstractTurnBuffer) = buffers(b).reward
terminal(b::AbstractTurnBuffer) = buffers(b).terminal

const RTSA = (:reward, :terminal, :state, :action)

Base.getindex(b::AbstractTurnBuffer{RTSA, types}, i::Int) where {types} = (
    state = state(b)[i],
    action = action(b)[i],
    reward = reward(b)[i+1],
    terminal = terminal(b)[i+1],
    next_state = state(b)[i+1],
    next_action = action(b)[i+1]
)

consecutive_view(b::AbstractTurnBuffer{RTSA}, inds, n) = NamedTuple{RTSA}(
    (state = consecutive_view(state(b), inds, n),
    action = consecutive_view(action(b), inds, n),
    reward = consecutive_view(reward(b), inds, n),
    terminal = consecutive_view(terminal(b), inds, n))
)


Base.length(b::AbstractTurnBuffer{RTSA}) = max(0, length(terminal(b)) - 1)