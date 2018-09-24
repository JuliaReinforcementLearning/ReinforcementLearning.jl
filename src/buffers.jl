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

import Base: (==), size, getindex, setindex!, length, push!, empty!, isempty, eltype, getproperty, view
mutable struct CircularArrayBuffer{A}
    capacity::Int
    first::Int
    length::Int
    stepsize::Int
    buffer::A

    CircularArrayBuffer{A}(capacity::Int, element_size::Vararg{Int, N}) where {A, N} = new{A}(capacity, 1, 0, *(element_size...), A(undef, element_size..., capacity))
    CircularArrayBuffer{A}(capacity::Int, element_size::Tuple{Vararg{Int, N}}) where {A, N} = CircularArrayBuffer{A}(capacity, element_size...)
end

function CircularArrayBuffer(capacity::Int, element::AbstractArray{T, N}) where {T, N}
    t = typeof(element)
    CircularArrayBuffer{t.name.wrapper{t.parameters[1], N+1}}(capacity, size(element))
end

@inline function _buffer_index(cb::CircularArrayBuffer, i::Int)
    n = cb.capacity
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
    start ≤ stop ? (start:stop) : vcat(start:cb.capacity, 1:stop)
end
Base.@propagate_inbounds function _buffer_index_checked(cb::CircularArrayBuffer, i::Int)
    @boundscheck if i < 1 || i > cb.length
        throw(BoundsError(cb, i))
    end
    _buffer_index(cb, i)
end

for func in [:view, :getindex]
    for N in 2:5
         @eval @__MODULE__() begin
             @inline Base.@propagate_inbounds $func(cb::CircularArrayBuffer{<:AbstractArray{T, $N}}, i::Int) where {T, N} = $func(cb.buffer, $(fill(Colon(), N-1)...),  _buffer_index_checked(cb, i))
             $func(cb::CircularArrayBuffer{<:AbstractArray{T, $N}}, i::UnitRange{Int}) where {T, N} = $func(cb.buffer, $(fill(Colon(), N-1)...),  _buffer_index(cb, i))
             $func(cb::CircularArrayBuffer{<:AbstractArray{T, $N}}, I::Vector{Int}) where {T, N} = $func(cb.buffer, $(fill(Colon(), N-1)...), [_buffer_index(cb, i) for i in I])
             $(Symbol(func, :consecutive))(cb::CircularArrayBuffer{<:AbstractArray{T, $N}}, I::Vector{Int}, n::Int) where {T, N} = reshape($func(cb, [j for i in I for j in i-n+1:i]), $([:(size(cb.buffer, $i)) for i in 1:N-2]...), n * size(cb.buffer, $(N-1)), length(I))
        end
    end
end

for N in 1:4
    @eval @__MODULE__() begin
        function getindexconsecutive(cb::CircularBuffer{<:AbstractArray{T, $N}}, I::Vector{Int}, n) where {T}
            elemsize = size(cb.buffer[1])
            stepsize = *(elemsize...)
            result = Array{Float64, $(N + 1)}(undef, $([:(elemsize[$i]) for i in 1:N-1]...), n * elemsize[$N], length(I))
            start = 1
            for i in I
                for j in n:-1:1
                    unsafe_copyto!(result, start, cb[i - j + 1], 1, stepsize)
                    start += stepsize
                end
            end
            result
        end
    end
end

"""
    push!(cb::CircularArrayBuffer{T, N}, data::E) where {T, N}

Add an element to the back and overwrite front if full.
Make sure that `length(data) == cb.stepsize`
"""
@inline function push!(cb::CircularArrayBuffer{<:AbstractArray{T, N1}}, data::AbstractArray{T, N2}) where {T, N1, N2}
    N1 == N2 + 1 || error("Cannot push data ot type $(typeof(data)) to buffer of type $(typeof(cb))")
#     length(data) == cb.stepsize || throw(DimensionMismatch("the length of buffer's stepsize doesn't match the length of data, $(cb.stepsize) != $(length(data))"))
    # if full, increment and overwrite, otherwise push
    if cb.length == cb.capacity
        cb.first = (cb.first == cb.capacity ? 1 : cb.first + 1)
    else
        cb.length += 1
    end
    nxt_idx = _buffer_index(cb, cb.length)
    @inbounds cb.buffer[cb.stepsize * (nxt_idx - 1) + 1: cb.stepsize * nxt_idx] = data
    cb
end

@inline function push!(cb::CircularArrayBuffer{<:AbstractArray{T, 1}}, data::T) where {T, N}
    if cb.length == cb.capacity
        cb.first = (cb.first == cb.capacity ? 1 : cb.first + 1)
    else
        cb.length += 1
    end
    @inbounds cb.buffer[_buffer_index(cb, cb.length)] = data
    cb
end

length(cb::CircularArrayBuffer) = cb.length
isfull(cb::CircularArrayBuffer) = length(cb) == capacity(cb)
isempty(cb::CircularArrayBuffer) = length(cb) == 0
empty!(cb::CircularArrayBuffer) = (cb.length = 0; cb)


"""
    ArrayCircularBuffer(arraytype, datatype, elemshape, capacity)
"""
function ArrayCircularBuffer(arraytype, datatype, elemshape, capacity)
    ArrayCircularBuffer(arraytype(zeros(datatype, 
                                        convert(Dims, (elemshape..., capacity)))),
                        capacity, 0, 0, false)
end
import Base.push!, Base.view, Base.lastindex, Base.getindex
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
