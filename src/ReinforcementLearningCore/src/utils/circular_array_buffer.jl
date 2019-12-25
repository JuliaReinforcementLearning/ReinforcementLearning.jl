export CircularArrayBuffer, capacity, isfull

mutable struct CircularArrayBuffer{T,N} <: AbstractArray{T,N}
    buffer::Array{T,N}
    first::Int
    length::Int
    step_size::Int

    function CircularArrayBuffer{T}(d::Integer...) where {T}
        N = length(d)
        new{T,N}(Array{T}(undef, d...), 1, 0, N == 1 ? 1 : *(d[1:end-1]...))
    end
end

Base.IndexStyle(::CircularArrayBuffer) = IndexLinear()
Base.size(cb::CircularArrayBuffer{<:Any, N}, i::Integer) where {N} = i == N ? cb.length : size(cb.buffer, i)
Base.size(cb::CircularArrayBuffer{<:Any,N}) where {N} = ntuple(M -> size(cb, M), N)
Base.getindex(cb::CircularArrayBuffer{T, N}, i::Int) where {T, N} = getindex(cb.buffer, _buffer_index(cb, i))
Base.setindex!(cb::CircularArrayBuffer{T, N}, v, i::Int) where {T, N} = setindex!(cb.buffer, v, _buffer_index(cb, i))
capacity(cb::CircularArrayBuffer{T, N}) where {T, N} = size(cb.buffer, N)
RLBase.isfull(cb::CircularArrayBuffer) = cb.length == capacity(cb)
Base.isempty(cb::CircularArrayBuffer) = cb.length == 0

@inline function _buffer_index(cb::CircularArrayBuffer, i::Int)
    ind = (cb.first - 1) * cb.step_size + i
    if ind > length(cb.buffer)
        ind - length(cb.buffer)
    else
        ind
    end
end

@inline function _buffer_frame(cb::CircularArrayBuffer, i::Int)
    n = capacity(cb)
    idx = cb.first + i - 1
    if idx > n
        idx - n
    else
        idx
    end
end

_buffer_frame(cb::CircularArrayBuffer, I::Vector{Int}) = map(i -> _buffer_frame(cb, i), I)

function Base.empty!(cb::CircularArrayBuffer)
    cb.first = 1
    cb.length = 0
    cb
end

function RLBase.update!(cb::CircularArrayBuffer{T,N}, data::AbstractArray{T}) where {T, N}
    select_last_dim(cb.buffer, _buffer_frame(cb, cb.length)) .= data
    cb
end

function RLBase.update!(cb::CircularArrayBuffer{T,1}, data::T) where {T}
    cb.buffer[_buffer_frame(cb, cb.length)] = data
    cb
end

function Base.push!(cb::CircularArrayBuffer, data)
    # length(data) == cb.step_size || throw(ArgumentError("length of , $(cb.step_size) != $(length(data))"))
    push!(cb, missing)
    update!(cb, data)
    cb
end

"""
!!! note
    When `push!`` a CircularArrayBuffer into a CircularArrayBuffer,
    only the last frame of the former is pushed!
"""
Base.push!(cb::CircularArrayBuffer{T, N}, data::CircularArrayBuffer{T, N}) where {T, N} = push!(cb, select_last_frame(data))

function Base.push!(cb::CircularArrayBuffer{T, N}, ::Missing) where {T, N}
    if cb.length == capacity(cb)
        cb.first = (cb.first == capacity(cb) ? 1 : cb.first + 1)
    else
        cb.length += 1
    end
    cb
end

# #####
# # gpu related
# ####

# if has_cuda()
#     using CuArrays
#     import Adapt:adapt

#     adapt(T::Type{<:CuArray}, x::SubArray{<:Any, <:Any, <:CircularArrayBuffer}) = T(x)
# end