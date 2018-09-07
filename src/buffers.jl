import DataStructures:isfull
import Base:size, getindex, setindex!, length, push!, empty!
abstract type AbstractBuffer{T, N} <: AbstractArray{T, N} end
abstract type AbstractSARDBuffer{Ts<:AbstractBuffer, Ta<:AbstractBuffer, Tr<:AbstractBuffer, Td<:AbstractBuffer} end

##############################
## CircularArrayBuffer
##############################

"Using a `N` dimension Array to simulate a `N-1` dimension circular buffer."
mutable struct CircularArrayBuffer{T, N} <: AbstractBuffer{T, N}
    buffer::Array{T, N}
    first::Int
    length::Int
    stepsize::Int
    CircularArrayBuffer{T}(capacity::Int, dims::Tuple{Vararg{Int}}) where T = new{T, length(dims)+1}(Array{T}(undef, dims..., capacity), 1, 0, *(dims...))
end

size(cb::CircularArrayBuffer{T, N}) where {T, N} = (size(cb.buffer)[1:N-1]..., cb.length)
getindex(cb::CircularArrayBuffer{T, N}, i::Int) where {T, N} = getindex(cb.buffer, [(:) for _ in 1 : N-1]...,  _buffer_index(cb, i))
getindex(cb::CircularArrayBuffer{T, N}, I::Vararg{Int, N}) where {T, N} = getindex(cb.buffer, I[1:N-1]...,  _buffer_index(cb, I[end]))
setindex!(cb::CircularArrayBuffer{T, N}, v, i::Int) where {T, N} = setindex!(cb.buffer, v, [(:) for _ in 1 : N-1]...,  _buffer_index(cb, i))
setindex!(cb::CircularArrayBuffer{T, N}, v, I::Vararg{Int, N}) where {T, N} = setindex!(cb.buffer, v, I[1:N-1]...,  _buffer_index(cb, I[end]))

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
@inline function push!(cb::CircularArrayBuffer, data)
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
## CircularSARDBuffer
##############################

struct CircularSARDBuffer{Ts, Ta, Tr, Td} <: AbstractSARDBuffer{Ts, Ta, Tr, Td}
    states::Ts
    actions::Ta
    rewards::Tr
    done::Td
end

function push!(b::CircularSARDBuffer{Ts, Ta, Tr, Td}, s::Ts, a::Ta, r::Tr, d::Td) where {Ts, Ta, Tr, Td}
    push!(b.states, s)
    push!(b.actions, a)
    push!(b.rewards, r)
    push!(b.done, d)
end

##############################
## EpisodeSARDBuffer
##############################

struct EpisodeSARDBuffer{Ts, Ta, Tr, Td} <: AbstractSARDBuffer{Ts, Ta, Tr, Td}
    states::Ts
    actions::Ta
    rewards::Tr
    done::Td
end

function push!(b::EpisodeSARDBuffer{Ts, Ta, Tr, Td}, s::Ts, a::Ta, r::Tr, d::Td) where {Ts, Ta, Tr, Td}
    if length(d) > 0 && convert(Bool, d[end])
        empty!(s); empty!(a); empty!(r); empty!(d)
    end
    push!(b.states, s); push!(b.actions, a); push!(b.rewards, r); push!(b.done, d); 
end

"""
    struct Buffer{Ts, Ta}
        states::CircularBuffer{Ts}
        actions::CircularBuffer{Ta}
        rewards::CircularBuffer{Float64}
        done::CircularBuffer{Bool}
"""
struct Buffer{Ts, Ta}
    states::CircularBuffer{Ts}
    actions::CircularBuffer{Ta}
    rewards::CircularBuffer{Float64}
    done::CircularBuffer{Bool}
end
"""
    Buffer(; statetype = Int64, actiontype = Int64, 
             capacity = 2, capacitystates = capacity,
             capacityrewards = capacity - 1)
"""
function Buffer(; statetype = Int64, actiontype = Int64, 
                  capacity = 2, capacitystates = capacity,
                  capacityrewards = capacity - 1)
    Buffer(CircularBuffer{statetype}(capacitystates),
           CircularBuffer{actiontype}(capacitystates),
           CircularBuffer{Float64}(capacityrewards),
           CircularBuffer{Bool}(capacityrewards))
end
function pushstateaction!(b, s, a)
    pushstate!(b, s)
    pushaction!(b, a)
end
pushstate!(b, s) = push!(b.states, deepcopy(s))
pushaction!(b, a) = push!(b.actions, a)
function pushreturn!(b, r, done)
    push!(b.rewards, r)
    push!(b.done, done)
end

"""
    struct EpisodeBuffer{Ts, Ta}
        states::Array{Ts, 1}
        actions::Array{Ta, 1}
        rewards::Array{Float64, 1}
        done::Array{Bool, 1}
"""
struct EpisodeBuffer{Ts, Ta}
    states::Array{Ts, 1}
    actions::Array{Ta, 1}
    rewards::Array{Float64, 1}
    done::Array{Bool, 1}
end
"""
    EpisodeBuffer(; statetype = Int64, actiontype = Int64) = 
        EpisodeBuffer(statetype[], actiontype[], Float64[], Bool[])
"""
EpisodeBuffer(; statetype = Int64, actiontype = Int64) = 
    EpisodeBuffer(statetype[], actiontype[], Float64[], Bool[])
function pushreturn!(b::EpisodeBuffer, r, done)
    if length(b.done) > 0 && b.done[end]
        s = b.states[end]; a = b.actions[end]
        empty!(b.states); empty!(b.actions); empty!(b.rewards); empty!(b.done)
        push!(b.states, s)
        push!(b.actions, a)
    end
    push!(b.rewards, r)
    push!(b.done, done)
end

"""
    mutable struct ArrayCircularBuffer{T}
        data::T
        capacity::Int64
        start::Int64
        counter::Int64
        full::Bool
"""
mutable struct ArrayCircularBuffer{T}
    data::T
    capacity::Int64
    start::Int64
    counter::Int64
    full::Bool
end
"""
    ArrayCircularBuffer(arraytype, datatype, elemshape, capacity)
"""
function ArrayCircularBuffer(arraytype, datatype, elemshape, capacity)
    ArrayCircularBuffer(arraytype(zeros(datatype, 
                                        convert(Dims, (elemshape..., capacity)))),
                        capacity, 0, 0, false)
end
import Base.push!, Base.view, Compat.lastindex, Base.getindex
for N in 2:5
    @eval @__MODULE__() begin
        function push!(a::ArrayCircularBuffer{<:AbstractArray{T, $N}}, x) where T
            n = prod(size(x))
            setindex!(a.data, x, a.counter*n + 1:(a.counter + 1)*n)
            a.counter += 1
            a.counter = a.counter % a.capacity
            if a.full 
                a.start += 1 
                a.start = a.start % a.capacity
            end
            if a.counter == 0 a.full = true end
            a.data
        end
    end
    for func in [:view, :getindex]
        @eval @__MODULE__() begin
            @inline function $func(a::ArrayCircularBuffer{<:AbstractArray{T, $N}}, i) where T
                idx = (a.start .+ i .- 1) .% a.capacity .+ 1
                $func(a.data, $(fill(Colon(), N-1)...), idx)
            end
            @inline function $(Symbol(:nmarkov, func))(a::ArrayCircularBuffer{<:AbstractArray{T, $N}}, i, nmarkov) where T
                nmarkov == 1 && return $func(a, i)
                numi = typeof(i) <: Number ? 1 : length(i)
                idx = zeros(Int64, numi*nmarkov)
                c = 1
                for j in i
                    for k in j - nmarkov + 1:j
                        idx[c] = (a.capacity + a.start + k - 1) % a.capacity + 1
                        c += 1
                    end
                end
                res = $func(a.data, $(fill(Colon(), N-1)...), idx)
                s = size(res)
                reshape(res, $([:(s[$x]) for x in 1:N-2]...), nmarkov * s[end-1], numi)
            end
        end
    end
end
lastindex(a::ArrayCircularBuffer) = a.full ? a.capacity : a.counter

"""
    struct ArrayStateBuffer{Ts, Ta}
        states::ArrayCircularBuffer{Ts}
        actions::CircularBuffer{Ta}
        rewards::CircularBuffer{Float64}
        done::CircularBuffer{Bool}
"""
struct ArrayStateBuffer{Ts, Ta}
    states::ArrayCircularBuffer{Ts}
    actions::CircularBuffer{Ta}
    rewards::CircularBuffer{Float64}
    done::CircularBuffer{Bool}
end
"""
    ArrayStateBuffer(; arraytype = Array, datatype = Float64, 
                       elemshape = (1), actiontype = Int64, 
                       capacity = 2, capacitystates = capacity,
                       capacityrewards = capacity - 1)

An `ArrayStateBuffer` is similar to a [`Buffer`](@ref) but the states are stored
in a prealocated array of size `(elemshape..., capacity)`. `K` consecutive
states at position `i` in the state buffer can can efficiently be retrieved with
`nmarkovview(buffer.states, i, K)` or `nmarkovgetindex(buffer.states, i, K)`.
See the implementation of DQN for an example. 
"""
function ArrayStateBuffer(; arraytype = Array, datatype = Float64, 
                            elemshape = (1), actiontype = Int64, 
                            capacity = 2, capacitystates = capacity,
                            capacityrewards = capacity - 1)
    ArrayStateBuffer(ArrayCircularBuffer(arraytype, datatype, elemshape, 
                                         capacitystates),
                     CircularBuffer{actiontype}(capacitystates),
                     CircularBuffer{Float64}(capacityrewards),
                     CircularBuffer{Bool}(capacityrewards))
end
pushstate!(b::ArrayStateBuffer, s) = push!(b.states, s)

import DataStructures.isfull
isfull(b::Union{Buffer, ArrayStateBuffer}) = isfull(b.rewards)
