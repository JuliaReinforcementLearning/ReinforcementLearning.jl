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

function generate_getindexconsecutive(A)
    for N in 1:4
        @eval @__MODULE__() begin
            function getindexconsecutive(cb::CircularBuffer{$A{T, $N}}, I, n) where {T}
                elemsize = size(cb.buffer[1])
                stepsize = *(elemsize...)
                result = $A{T, $(N + 1)}(undef, $([:(elemsize[$i]) for i in 1:N-1]...), n * elemsize[$N], length(I))
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
end
generate_getindexconsecutive(Array)

import DataStructures: _buffer_index
@inline function _buffer_index(cb::CircularBuffer, i::Int)
    n = cb.capacity
    idx = cb.first + i - 1
    idx > 0 && idx <= n && return idx
    idx <= 0 && return n + idx % n
    (idx - 1) % n + 1
end

state(buffer) = buffer.states[length(buffer.states) - 1]
states(buffer) = buffer.states[1:length(buffer.states) - 1]
states(buffer, I) = buffer.states[I]
statesconsecutive(buffer, I, n) = getindexconsecutive(buffer.states, I, n)
nextstate(buffer) = buffer.states[length(buffer.states)]
nextstates(buffer) = buffer.states[2:length(buffer.states)]
neststates(buffer) = buffer.states[I .+ 1]
nextstatesconsecutive(buffer, I, n) = getindexconsecutive(buffer.states, I .+ 1, n)
action(buffer) = buffer.actions[length(buffer.actions) - 1]
nextaction(buffer) = buffer.actions[length(buffer.actions)]

import DataStructures.isfull
isfull(b::Buffer) = isfull(b.done)
