export ReservoirArrayBuffer

using Random
using ElasticArrays
using MacroTools: @forward

mutable struct ReservoirArrayBuffer{T,N,B<:ElasticArray{T,N},R<:AbstractRNG} <:
               AbstractArray{T,N}
    buffer::B
    n::Int
    capacity::Int
    rng::R
end

ReservoirArrayBuffer{T}(dims::Int...; rng = Random.GLOBAL_RNG) where {T} =
    ReservoirArrayBuffer(ElasticArray{T}(undef, dims[1:end-1]..., 0), 0, dims[end], rng)

@forward ReservoirArrayBuffer.buffer Base.size,
Base.getindex,
Base.length,
Base.sizeof,
Base.IndexStyle

# TODO: rename all push! to append!

function Base.push!(b::ReservoirArrayBuffer{T,N}, x) where {T,N}
    b.n += 1
    if b.n <= b.capacity
        append!(b.buffer, x)
    else
        i = rand(b.rng, 1:b.n)
        if i <= b.capacity
            stride = b.buffer.kernel_length.divisor
            b.buffer.data[(stride*(i-1)+1):stride*i] .= x
        end
    end
end
