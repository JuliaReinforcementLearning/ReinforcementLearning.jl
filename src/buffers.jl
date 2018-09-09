import Base:size, getindex, setindex!, length, push!, empty!, isempty

abstract type AbstractBuffer{T, N} <: AbstractArray{T, N} end
abstract type AbstractTurn end
abstract type AbstractTurnBuffer{T<:AbstractTurn} <: AbstractArray{T, 1} end

##############################
## Turn
##############################

struct Turn{Ts, Ta, Tr, Td} <: AbstractTurn
    state::Ts
    action::Ta
    reward::Tr
    isdone::Td
    nextstate::Ts
end


##############################
## CircularArrayBuffer
##############################

"""
    CircularArrayBuffer{T}(capacity::Int, element_size::Tuple{Vararg{Int, N}})

Using a `N` dimension Array to simulate a `N-1` dimension circular buffer.
"""
mutable struct CircularArrayBuffer{E, T, N} <: AbstractBuffer{T, N}
    buffer::Array{T, N}
    first::Int
    length::Int
    stepsize::Int
    CircularArrayBuffer{T}(capacity::Int, element_size::Tuple{Vararg{Int, N}}) where {T,N} = new{Array{T, N}, T, N+1}(Array{T, N+1}(undef, element_size..., capacity), 1, 0, *(element_size...))
end

size(cb::CircularArrayBuffer{E, T, N}) where {E, T, N} = (size(cb.buffer)[1:N-1]..., cb.length)
getindex(cb::CircularArrayBuffer{E, T, N}, i::Int) where {E, T, N} = getindex(cb.buffer, [(:) for _ in 1 : N-1]...,  _buffer_index(cb, i))
getindex(cb::CircularArrayBuffer{E, T, N}, I::Vararg{Int, N}) where {E, T, N} = getindex(cb.buffer, I[1:N-1]...,  _buffer_index(cb, I[end]))
setindex!(cb::CircularArrayBuffer{E, T, N}, v, i::Int) where {E, T, N} = setindex!(cb.buffer, v, [(:) for _ in 1 : N-1]...,  _buffer_index(cb, i))
setindex!(cb::CircularArrayBuffer{E, T, N}, v, I::Vararg{Int, N}) where {E, T, N} = setindex!(cb.buffer, v, I[1:N-1]...,  _buffer_index(cb, I[end]))

capacity(cb::CircularArrayBuffer) = size(cb.buffer)[end]
length(cb::CircularArrayBuffer) = cb.length
isfull(cb::CircularArrayBuffer) = length(cb) == capacity(cb)
isempty(cb::CircularArrayBuffer) = length(cb) == 0
empty!(cb::CircularArrayBuffer) = (cb.length = 0; cb)

"""
    push!(cb::CircularArrayBuffer{E, T, N}, data::E)

Add an element to the back and overwrite front if full.
Make sure that `length(data) == cb.stepsize`
"""
@inline function push!(cb::CircularArrayBuffer{E, T, N}, data::E) where {E, T, N}
    # @boundscheck length(data) == cb.stepsize
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

@inline function _buffer_index(cb::CircularArrayBuffer, i::Int)
    n = capacity(cb)
    idx = cb.first + i - 1
    if idx > n
        idx - n
    else
        idx
    end
end


##############################
## CircularTurnBuffer
##############################

struct CircularTurnBuffer{Ts, Ta, Tr, Td} <: AbstractTurnBuffer{Turn{Ts, Ta, Tr, Td}}
    states::CircularArrayBuffer{Ts}
    actions::CircularArrayBuffer{Ta}
    rewards::CircularArrayBuffer{Tr}
    isdone::CircularArrayBuffer{Td}
    nextstates::CircularArrayBuffer{Ts}
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


##############################

isempty(b::AbstractTurnBuffer{Turn}) = isempty(b.states)  # check `states` field is enough

function empty!(b::AbstractTurnBuffer{Turn})
    empty!(b.states); empty!(b.actions); empty!(b.rewards); empty!(b.isdone); empty!(b.nextstates)
end
