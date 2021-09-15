export buffered_shuffle
export RLTransition
export batch!
export RingBuffer
export gen_JuliaRL_dataset

import Base.take!
import Base.iterate

using Random
using ProgressMeter

abstract type RLTransition end

#####
#  BufferedShuffle
#####
"""
    BufferedShuffle(src::Channel{T}, buffer::Vector{T}, rng<:AbstractRNG)

This type holds the output of `buffered_shuffle` function and subtypes AbstractChannel. 
Therefore, it acts as a channel that holds a shuffled buffer which is of type Vector{T}.

## Fields

- `src::Channel{T}`, The source `Channel`.
- `buffer::Vector{T}`, The shuffled buffer.
- `rng<:AbstractRNG`.
"""
struct BufferedShuffle{T, R<:AbstractRNG} <: AbstractChannel{T}
    src::Channel{T}
    buffer::Vector{T}
    rng::R
end

"""
    buffered_shuffle(src::Channel{T}, buffer_size::Int; rng=Random.GLOBAL_RNG)

Returns a `BufferedShuffle` `Channel`. 

Arguments:
- `src::Channel{T}`. The source Channel.
- `buffer_size::Int`. The size of the buffered channel.
- `rng<:AbstractRNG` = Random.GLOBAL_RNG.
"""
function buffered_shuffle(src::Channel{T}, buffer_size::Int;rng=Random.GLOBAL_RNG) where T
    buffer = Array{T}(undef, buffer_size)
    p = Progress(buffer_size)
    Threads.@threads for i in 1:buffer_size
        buffer[i] = take!(src)
        next!(p)
    end
    BufferedShuffle(src, buffer, rng)
end

Base.close(b::BufferedShuffle) = close(b.src)

function Base.take!(b::BufferedShuffle)
    if length(b.buffer) == 0
        throw(InvalidStateException("buffer is empty", :empty))
    else
        i = rand(b.rng, 1:length(b.buffer))
        res = b.buffer[i]
        if isopen(b.src)
            b.buffer[i] = popfirst!(b.src)
        else
            deleteat!(b.buffer, i)
        end
        res
    end
end

function Base.iterate(b::BufferedShuffle, state=nothing)
    try
        return (popfirst!(b), nothing)
    catch e
        if isa(e, InvalidStateException) && e.state === :empty
            return nothing
        else
            rethrow()
        end
    end
end

#####
# RingBuffer
#####

mutable struct RingBuffer{T} <: AbstractChannel{T}
    buffers::Channel{T}
    current::T
    results::Channel{T}
end

Base.close(b::RingBuffer) = close(b.buffers) # will propergate to b.results
"""
    RingBuffer(f!, buffer, taskref=nothing)

Return a RingBuffer that gives batches with the specs in `buffer`.

# Arguments

- `f!`: the inplace operation to do in the `buffer`.
- `buffer::T`: the type containing the batch.
- `sz::Int`:size of the internal buffers.
"""
function RingBuffer(f!, buffer::T;sz=Threads.nthreads(), taskref=nothing) where T
    buffers = Channel{T}(sz)
    for _ in 1:sz
        put!(buffers, deepcopy(buffer))
    end
    results = Channel{T}(sz, spawn=true, taskref=taskref) do ch
        Threads.foreach(buffers;schedule=Threads.StaticSchedule()) do x
        # for x in buffers
            f!(x)  # in-place operation
            put!(ch, x)
        end
    end
    RingBuffer(buffers, buffer, results)
end

function Base.take!(b::RingBuffer)
    put!(b.buffers, b.current)
    b.current = take!(b.results)
    b.current
end

function batch!(dest::RLTransition, src::RLTransition, i::Int)
    for fn in fieldnames(typeof(dest))
        xs = getfield(dest, fn)
        x = getfield(src, fn)
        selectdim(xs, ndims(xs), i) .= x
    end
end
