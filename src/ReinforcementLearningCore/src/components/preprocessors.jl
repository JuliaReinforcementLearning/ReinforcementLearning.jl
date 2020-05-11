export StackFrames, ResizeImage

using ImageTransformations: imresize!

"""
    ResizeImage(img::Array{T, N})
    ResizeImage(dims::Int...) -> ResizeImage(Float32, dims...)
    ResizeImage(T::Type{<:Number}, dims::Int...)

Using BSpline method to resize the `state` field of an observation to size of `img` (or `dims`).
"""
struct ResizeImage{T,N} <: AbstractPreprocessor
    img::Array{T,N}
end

ResizeImage(dims::Int...) = ResizeImage(Float32, dims...)
ResizeImage(T::Type{<:Number}, dims::Int...) = ResizeImage(Array{T}(undef, dims))

function (p::ResizeImage)(state::AbstractArray)
    imresize!(p.img, state)
    p.img
end

"""
    StackFrames(::Type{T}=Float32, d::Int...)

Use a pre-initialized [`CircularArrayBuffer`](@ref) to store the latest several states specified by `d`. Before processing any observation, the buffer is filled with `zero{T}`.
"""
struct StackFrames{T,N} <: AbstractPreprocessor
    buffer::CircularArrayBuffer{T,N}
end

StackFrames(d::Int...) = StackFrames(Float32, d...)

function StackFrames(::Type{T}, d::Vararg{Int,N}) where {T,N}
    p = StackFrames(CircularArrayBuffer{T}(d...))
    for _ in 1:capacity(p.buffer)
        push!(p.buffer, zeros(T, size(p.buffer)[1:N-1]))
    end
    p
end

function (p::StackFrames{T,N})(state::AbstractArray) where {T,N}
    push!(p.buffer, state)
    p.buffer
end

# !!! side effect?
function Base.push!(
    cb::CircularArrayBuffer{T,N},
    stacked_data::CircularArrayBuffer{T,N},
) where {T,N}
    push!(cb, select_last_frame(stacked_data))
end
