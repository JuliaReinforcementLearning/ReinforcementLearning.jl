import Base: (==), size, getindex, setindex!, length, push!, empty!, isempty, eltype

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

==(t1::Turn, t2::Turn) = t1.state == t2.state &&
                           t1.action == t2.action && 
                           t1.reward == t2.reward && 
                           t1.isdone == t2.isdone && 
                           t1.nextstate == t2.nextstate


##############################
## CircularArrayBuffer
##############################

"""
    CircularArrayBuffer{T}(capacity::Int, element_size::Tuple{Vararg{Int, N}}) where {T,N}

Using a `N` dimension Array to simulate a `N-1` dimension circular buffer.
Used in [`CircularTurnBuffer`](@ref).
"""
mutable struct CircularArrayBuffer{E, T, N} <: AbstractBuffer{T, N}
    buffer::Array{T, N}
    first::Int
    length::Int
    stepsize::Int
    CircularArrayBuffer{T}(capacity::Int) where T<:Number = new{T, T, 1}(Vector{T}(undef, capacity), 1, 0, 1)
    function CircularArrayBuffer{T}(capacity::Int, element_size::Vararg{Int, N}) where {T<:AbstractArray,N} 
        ndims(T) == N || throw(DimensionMismatch("the ndims of the specified type $T doesn't math the length of element_size $element_size"))
        new{T, eltype(T), N+1}(Array{eltype(T), N+1}(undef, element_size..., capacity), 1, 0, *(element_size...))
    end
    CircularArrayBuffer{T}(capacity::Int, element_size::Tuple{Vararg{Int, N}}) where {T<:AbstractArray,N} = CircularArrayBuffer{T}(capacity, element_size...)
end

eltype(cb::CircularArrayBuffer{E, T, N}) where {E, T, N} = E
size(cb::CircularArrayBuffer{E, T, N}) where {E, T, N} = (size(cb.buffer)[1:N-1]..., cb.length)

getindex(cb::CircularArrayBuffer{E, T, N}, i::Int) where {E, T, N} = getindex(cb.buffer, [(:) for _ in 1 : N-1]...,  _buffer_index(cb, i))
getindex(cb::CircularArrayBuffer{E, T, N}, i::UnitRange{Int}) where {E, T, N} = getindex(cb.buffer, [(:) for _ in 1 : N-1]...,  _buffer_index(cb, i))
getindex(cb::CircularArrayBuffer{E, T, N}, I::Vector{Int}) where {E, T, N} = getindex(cb.buffer, [(:) for _ in 1 : N-1]..., [_buffer_index(cb, i) for i in I])

## kept only to show elements
## never manually get/set the inner elements because it is very slow
## see https://discourse.julialang.org/t/varargs-performance/13578
getindex(cb::CircularArrayBuffer{E, T, N}, I::Vararg{Int, N}) where {E, T, N} = getindex(cb.buffer, I[1:N-1]...,  _buffer_index(cb, I[end]))

"""
    getconsecutive(cb::CircularArrayBuffer{E, T, N}, i::Int, n::Int)

Get the `n` consecutive elements in the buffer before `i`(`i`-th element is included).
`i` must greater than `n`.
"""
getconsecutive(cb::CircularArrayBuffer{E, T, N}, i::Int, n::Int) where {E,T,N} =  getindex(cb, i-n+1:i)

"""
    getconsecutive(cb::CircularArrayBuffer{E, T, N}, I::Vector{Int}, n::Int)

Get the `n` consecutive elements in the buffer before each element in `I`.
Each element in `I` must greater than `n`.
Return an `Array{T, N+1}`
"""
getconsecutive(cb::CircularArrayBuffer{E, T, N}, I::Vector{Int}, n::Int) where {E,T,N} = reshape(getindex(cb, [j for i in I for j in i-n+1:i]), size(cb.buffer)[1:N-1]..., n, length(I))

capacity(cb::CircularArrayBuffer) = size(cb.buffer)[end]
length(cb::CircularArrayBuffer) = cb.length
isfull(cb::CircularArrayBuffer) = length(cb) == capacity(cb)
isempty(cb::CircularArrayBuffer) = length(cb) == 0
empty!(cb::CircularArrayBuffer) = (cb.length = 0; cb)

"""
    push!(cb::CircularArrayBuffer{E, T, N}, data::E) where {E, T, N}

Add an element to the back and overwrite front if full.
Make sure that `length(data) == cb.stepsize`
"""
@inline function push!(cb::CircularArrayBuffer{E, T, N}, data::AbstractArray{T}) where {E, T, N}
    length(data) == cb.stepsize || throw(DimensionMismatch("the length of buffer's stepsize doesn't match the length of data, $(cb.stepsize) != $(length(data))"))
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

@inline function push!(cb::CircularArrayBuffer{E, T, 1}, data::T) where {E, T}
    if cb.length == capacity(cb)
        cb.first = (cb.first == capacity(cb) ? 1 : cb.first + 1)
    else
        cb.length += 1
    end
    cb.buffer[_buffer_index(cb, cb.length)] = data
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

@inline function _buffer_index(cb::CircularArrayBuffer, i::UnitRange{Int})
    start = _buffer_index(cb, i.start)
    stop = _buffer_index(cb, i.stop)
    start ≤ stop ? (start:stop) : vcat(start:capacity(cb), 1:stop)
end

##############################
## CircularTurnBuffer
##############################

"""
    CircularTurnBuffer{Ts, Ta, Tr, Td}(capacity::Int,
                                       size_s::Tuple{Vararg{Int}},
                                       size_a::Tuple{Vararg{Int}},
                                       size_r::Tuple{Vararg{Int}},
                                       size_d::Tuple{Vararg{Int}},
                                       size_ns::Tuple{Vararg{Int}}) 

Store the [`Turn`](@ref) info into a circular buffer of `capacity`.

See also: [`EpisodeTurnBuffer`](@ref)
"""
struct CircularTurnBuffer{Ts, Ta, Tr, Td} <: AbstractTurnBuffer{Turn{Ts, Ta, Tr, Td}}
    states::CircularArrayBuffer{Ts}
    actions::CircularArrayBuffer{Ta}
    rewards::CircularArrayBuffer{Tr}
    isdone::CircularArrayBuffer{Td}
    nextstates::CircularArrayBuffer{Ts}
    function CircularTurnBuffer{Ts, Ta, Tr, Td}(
        capacity::Int,
        size_s::Tuple{Vararg{Int}}=(),
        size_a::Tuple{Vararg{Int}}=(),
        size_r::Tuple{Vararg{Int}}=(),
        size_d::Tuple{Vararg{Int}}=()) where {Ts, Ta, Tr, Td}
        new{Ts, Ta, Tr, Td}(
            CircularArrayBuffer{Ts}(capacity, size_s...),
            CircularArrayBuffer{Ta}(capacity, size_a...),
            CircularArrayBuffer{Tr}(capacity, size_r...),
            CircularArrayBuffer{Td}(capacity, size_d...),
            CircularArrayBuffer{Ts}(capacity, size_s...))
    end
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

"""
    EpisodeTurnBuffer{Ts, Ta, Tr, Td}()

Store the [`Turn`](@ref) info into a buffer.
The only difference with [`CircularTurnBuffer`](@ref) is that,
while `push!` a `Turn` info the `EpisodeTurnBuffer`,
the buffer is emptied first if last turn is the end of an episode

See also: [`CircularTurnBuffer`](@ref)
"""
struct EpisodeTurnBuffer{Ts, Ta, Tr, Td} <: AbstractTurnBuffer{Turn{Ts, Ta, Tr, Td}}
    states::Vector{Ts}
    actions::Vector{Ta}
    rewards::Vector{Tr}
    isdone::Vector{Td}
    nextstates::Vector{Ts}
    EpisodeTurnBuffer{Ts, Ta, Tr, Td}() where {Ts, Ta, Tr, Td} = new(Ts[], Ta[], Tr[], Td[], Ts[])
end

function push!(b::EpisodeTurnBuffer{Ts, Ta, Tr, Td}, t::Turn{Ts, Ta, Tr, Td}) where {Ts, Ta, Tr, Td}
    if !isempty(b) && convert(Bool, b.isdone[end]) # last turn is the end of an episode
        empty!(b)
    end
    push!(b.states, t.state)
    push!(b.actions, t.action)
    push!(b.rewards, t.reward)
    push!(b.isdone, t.isdone)
    push!(b.nextstates, t.nextstate)
end


##############################
getconsecutive(v::Vector, I::Vector{Int}, n::Int) = reshape(v[[x for i in I for x in i-n+1:i]], n, length(I))

size(b::AbstractTurnBuffer{<:Turn}) = (length(b.states),)
length(b::AbstractTurnBuffer{<:Turn}) = length(b.states)
eltype(b::AbstractTurnBuffer{Turn{Ts, Ta, Tr, Td}}) where {Ts, Ta, Tr, Td} = Turn{Ts, Ta, Tr, Td}
isempty(b::AbstractTurnBuffer{<:Turn}) = isempty(b.states)  # check `states` field is enough

getindex(b::AbstractTurnBuffer{<:Turn}, i::Int) = Turn(b.states[i], b.actions[i], b.rewards[i], b.isdone[i], b.nextstates[i])
getconsecutive(b::AbstractTurnBuffer{<:Turn}, i::Int, n::Int) = Turn(b.states[i-n+1:i], b.actions[i-n+1:i], b.rewards[i-n+1:i], b.isdone[i-n+1:i], b.nextstates[i-n+1:i])
getconsecutive(b::AbstractTurnBuffer{<:Turn}, I::Vector{Int}, n::Int) = Turn(
    getconsecutive(b.states, I, n),
    getconsecutive(b.actions, I, n),
    getconsecutive(b.rewards, I, n),
    getconsecutive(b.isdone, I, n),
    getconsecutive(b.nextstates, I, n))

function empty!(b::AbstractTurnBuffer{<:Turn})
    empty!(b.states); empty!(b.actions); empty!(b.rewards); empty!(b.isdone); empty!(b.nextstates)
end

function getindex(b::AbstractTurnBuffer{<:Turn}, i::Int)
    Turn(b.states[i], b.actions[i], b.rewards[i], b.isdone[i], b.nextstates[i])
end