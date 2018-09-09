import DataStructures:isfull
import Base:size, getindex, setindex!, length, push!, empty!, isempty

abstract type AbstractBuffer{T, N} <: AbstractArray{T, N} end
abstract type AbstractTurn end
abstract type AbstractTurnBuffer{T<:AbstractTurn} <: AbstractArray{T, 1} end

############################################################
## Turn
############################################################

struct Turn{Ts, Ta, Tr, Td} <: AbstractTurn
    state::Ts
    action::Ta
    reward::Tr
    isdone::Td
    nextstate::Ts
end


############################################################
## CircularArrayBuffer
############################################################

"Using a `N` dimension Array to simulate a `N-1` dimension circular buffer."
mutable struct CircularArrayBuffer{E<:AbstractArray, T, N} <: AbstractBuffer{T, N}
    buffer::AbstractArray{T, N}
    first::Int
    length::Int
    stepsize::Int
    CircularArrayBuffer{E}(capacity::Int, size::Tuple{Int}) where E<:AbstractArray = new{E, eltype(E), ndims(E)+1}(E(undef, size..., capacity), 1, 0, *(size...))
end

size(cb::CircularArrayBuffer{E, T, N}) where {E, T, N} = (size(cb.buffer)[1:N-1]..., cb.length)
getindex(cb::CircularArrayBuffer{E, T, N}, i::Int) where {E, T, N} = getindex(cb.buffer, [(:) for _ in 1 : N-1]...,  _buffer_index(cb, i))
getindex(cb::CircularArrayBuffer{E, T, N}, I::Vararg{Int, N}) where {E, T, N} = getindex(cb.buffer, I[1:N-1]...,  _buffer_index(cb, I[end]))
setindex!(cb::CircularArrayBuffer{E, T, N}, v, i::Int) where {E, T, N} = setindex!(cb.buffer, v, [(:) for _ in 1 : N-1]...,  _buffer_index(cb, i))
setindex!(cb::CircularArrayBuffer{E, T, N}, v, I::Vararg{Int, N}) where {E, T, N} = setindex!(cb.buffer, v, I[1:N-1]...,  _buffer_index(cb, I[end]))

""""
    capacity(cb)
Return capacity of CircularArrayBuffer.
"""
capacity(cb::CircularArrayBuffer) = size(cb.buffer)[end]

"""
    length(cb)
Return the number of elements currently in the buffer.
"""
length(cb::CircularArrayBuffer) = cb.length

"""
    isfull(cb)
Test whether the buffer is full.
"""
isfull(cb::CircularArrayBuffer) = length(cb) == capacity(cb)

"""
    push!(cb)
Add an element to the back and overwrite front if full.
"""
@inline function push!(cb::CircularArrayBuffer{E, T, N}, data::E) where {E, T, N}
    @boundscheck length(data) == cb.stepsize
    # if full, increment and overwrite, otherwise push
    if cb.length == capacity(cb)
        cb.first = (cb.first == capacity(cb) ? 1 : cb.first + 1)
    else
        cb.length += 1
    end
    nxt_idx = _buffer_index(cb, cb.length)
    cb.buffer[cb.stepsize * (nxt_idx - 1) + 1: cb.stepsize * nxt_idx] = data
    cb
end

"""
    empty!(cb)
Reset the buffer.
"""
function empty!(cb::CircularArrayBuffer)
    cb.length = 0
    cb
end

"""
    isempty(cb)
Check is the buffer empty.
"""
isempty(cb::CircularArrayBuffer) = length(cb) == 0

@inline function _buffer_index(cb::CircularArrayBuffer, i::Int)
    n = capacity(cb)
    idx = cb.first + i - 1
    if idx > n
        idx - n
    else
        idx
    end
end


############################################################
## CircularTurnBuffer
############################################################

struct CircularTurnBuffer{Ts, Ta, Tr, Td} <: AbstractTurnBuffer{Turn{Ts, Ta, Tr, Td}}
    states::CircularArrayBuffer{Ts}
    actions::CircularArrayBuffer{Ta}
    rewards::CircularArrayBuffer{Tr}
    isdone::CircularArrayBuffer{Td}
    nextstates::CircularArrayBuffer{Ts}
end

isempty(b::CircularTurnBuffer) = isempty(b.states)  # check `states` field is enough

function empty!(b::CircularTurnBuffer)
    empty!(b.states); empty!(b.actions); empty!(b.rewards); empty!(b.isdone); empty!(b.nextstates)
end

function push!(b::CircularTurnBuffer{Ts, Ta, Tr, Td}, t::Turn{Ts, Ta, Tr, Td}) where {Ts, Ta, Tr, Td}
    push!(b.states, t.state)
    push!(b.actions, t.action)
    push!(b.rewards, t.reward)
    push!(b.isdone, t.isdone)
    push!(b.nextstates, t.nextstate)
end

##############################
## EpisodeTurnBuffer
##############################

struct EpisodeTurnBuffer{Ts, Ta, Tr, Td} <: AbstractTurnBuffer{Turn{Ts, Ta, Tr, Td}}
    states::CircularArrayBuffer{Ts}
    actions::CircularArrayBuffer{Ta}
    rewards::CircularArrayBuffer{Tr}
    isdone::CircularArrayBuffer{Td}
    nextstates::CircularArrayBuffer{Ts}
end

isempty(b::EpisodeTurnBuffer) = isempty(b.states)  # check `states` field is enough

function empty!(b::EpisodeTurnBuffer)
    empty!(b.states); empty!(b.actions); empty!(b.rewards); empty!(b.isdone); empty!(b.nextstates)
end

function push!(b::EpisodeTurnBuffer{Ts, Ta, Tr, Td}, t::Turn{Ts, Ta, Tr, Td}) where {Ts, Ta, Tr, Td}
    if !isempty(b) && convert(Bool, d[end]) # last turn is the end of an episode
        empty!(b)
    end
    push!(b.states, t.state)
    push!(b.actions, t.action)
    push!(b.rewards, t.reward)
    push!(b.isdone, t.isdone)
    push!(b.nextstates, t.nextstate)
end
