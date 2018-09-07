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
