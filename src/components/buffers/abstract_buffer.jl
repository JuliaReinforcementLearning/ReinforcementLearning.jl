export AbstractTurnBuffer, buffers, RTSA, isfull

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

const RTSA = (:reward, :terminal, :state, :action)

Base.getindex(b::AbstractTurnBuffer{RTSA, types}, i::Int) where {types} = (
    state = buffers(b).state[i],
    action = buffers(b).action[i],
    reward = buffers(b).reward[i+1],
    terminal = buffers(b).terminal[i+1],
    next_state = buffers(b).state[i+1],
    next_action = buffers(b).action[i+1]
)

Base.length(b::AbstractTurnBuffer{RTSA}) = max(0, length(buffers(b).terminal) - 1)