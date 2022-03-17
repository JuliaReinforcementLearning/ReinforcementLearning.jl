export StackFrames

import CircularArrayBuffers
using CircularArrayBuffers: CircularArrayBuffer
using MacroTools: @forward

"""
    StackFrames(::Type{T}=Float32, d::Int...)

Use a pre-initialized [`CircularArrayBuffer`](@ref) to store the latest several states specified by `d`. Before processing any observation, the buffer is filled with `zero{T}
by default.
"""
struct StackFrames{T,N} <: AbstractArray{T,N}
    buffer::CircularArrayBuffer{T,N}
end

@forward StackFrames.buffer Base.size, Base.getindex
Base.IndexStyle(x::StackFrames) = IndexStyle(x.buffer)

StackFrames(d::Int...) = StackFrames(Float32, d...)

function StackFrames(::Type{T}, d::Vararg{Int,N}) where {T,N}
    b = CircularArrayBuffer{T}(d...)
    append!(b, zeros(T, d...))
    StackFrames(b)
end

function (p::StackFrames{T,N})(state::AbstractArray) where {T,N}
    push!(p.buffer, state)
    p
end

function RLBase.reset!(p::StackFrames{T,N}) where {T,N}
    fill!(p.buffer, zero(T))
    p
end

"""
When pushing a `StackFrames` into a `CircularArrayBuffer` of the same dimension,
only the latest frame is pushed. If the `StackFrames` is one dimension lower,
then it is treated as a general `AbstractArray` and is pushed in as a frame.
"""
function Base.push!(cb::CircularArrayBuffer{T,N}, p::StackFrames{T,N}) where {T,N}
    push!(cb, select_last_frame(p.buffer))
end
