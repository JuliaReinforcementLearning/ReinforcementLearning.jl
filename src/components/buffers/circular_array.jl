export CircularArray

"""
    CircularArray(T, d...)

Similar to [`CircularArrayBuffer`](@ref), but faster when accessing inner elements.
"""
mutable struct CircularArray{T, N} <: AbstractArray{T, N}
    frames::Array{T, N}
    cursor::Int
    stride::Int
    CircularArray(::Type{T}, I::Vararg{Int, N}) where {T, N} = new{T, N}(zeros(T, I), 1, *(I[1:end-1]...))
end

function Base.push!(A::CircularArray{T, N}, data::AbstractArray) where {T, N}
    selectdim(A.frames, N, A.cursor) .= data

    A.cursor += 1
    if A.cursor > size(A.frames, N)
        A.cursor = 1
    end
end

Base.IndexStyle(::CircularArray) = IndexLinear()
Base.size(A::CircularArray) = size(A.frames)

function Base.getindex(A::CircularArray, i::Int)
    ind = i + A.stride * (A.cursor - 1)
    n = length(A.frames)
    if ind > n
        ind -= n
    end
    A.frames[ind]
end