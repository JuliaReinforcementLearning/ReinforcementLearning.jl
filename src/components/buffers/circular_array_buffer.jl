export CircularArrayBuffer, capacity, isfull, consecutive_view, select_frame

mutable struct CircularArrayBuffer{T,N} <: AbstractArray{T,N}
    buffer::Array{T,N}
    first::Int
    length::Int
    step_size::Int

    function CircularArrayBuffer{T}(d::Integer...) where {T}
        N = length(d)
        N > 0 || throw(ArgumentError("dimension must be greater than 0"))
        new{T,N}(Array{T}(undef, d...), 1, 0, N == 1 ? 1 : *(d[1:end-1]...))
    end
end

Base.IndexStyle(::CircularArrayBuffer) = IndexLinear()
Base.size(cb::CircularArrayBuffer{T,N}) where {T,N} = (size(cb.buffer)[1:N-1]..., cb.length)
Base.getindex(cb::CircularArrayBuffer{T, N}, i::Int) where {T, N} = getindex(cb.buffer, _buffer_index(cb, i))
Base.setindex!(cb::CircularArrayBuffer{T, N}, v, i::Int) where {T, N} = setindex!(cb.buffer, v, _buffer_index(cb, i))
capacity(cb::CircularArrayBuffer{T, N}) where {T, N} = size(cb.buffer, N)
isfull(cb::CircularArrayBuffer) = cb.length == capacity(cb)
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

function Base.push!(cb::CircularArrayBuffer{T,N}, data) where {T,N}
    # length(data) == cb.step_size || throw(ArgumentError("length of , $(cb.step_size) != $(length(data))"))
    if cb.length == capacity(cb)
        cb.first = (cb.first == capacity(cb) ? 1 : cb.first + 1)
    else
        cb.length += 1
    end
    selectdim(cb.buffer, N, _buffer_frame(cb, cb.length)) .= data
    cb
end

Base.push!(cb::CircularArrayBuffer{T, N}, data::CircularArrayBuffer{T, N}) where {T, N} = push!(cb, select_frame(data, data.length))

select_frame(cb::CircularArrayBuffer{T, 1}, i) where {T} = getindex(cb.buffer, _buffer_frame(cb, i))
select_frame(cb::CircularArrayBuffer{T, 2}, i) where {T} = view(cb.buffer, :, _buffer_frame(cb, i))
select_frame(cb::CircularArrayBuffer{T, 3}, i) where {T} = view(cb.buffer, :, :, _buffer_frame(cb, i))
select_frame(cb::CircularArrayBuffer{T, 4}, i) where {T} = view(cb.buffer, :, :, :, _buffer_frame(cb, i))

select_frame(cb::Array{T, 1}, i::Int) where {T} = getindex(cb, i)
select_frame(cb::Array{T, N}, i::Int) where {T, N} = selectdim(cb, N, i)

consecutive_view(cb::CircularArrayBuffer, inds::Vector{Int}) = select_frame(cb, inds)

consecutive_view(cb::CircularArrayBuffer{T,1}, inds::Vector{Int}, n::Int) where {T} = reshape(view(cb.buffer, [_buffer_frame(cb, i) for x in inds for i in x:x+n-1]), n, length(inds))
consecutive_view(cb::CircularArrayBuffer{T,2}, inds::Vector{Int}, n::Int) where {T} = reshape(view(cb.buffer, :, [_buffer_frame(cb, i) for x in inds for i in x:x+n-1]), size(cb.buffer, 1), n, length(inds))
consecutive_view(cb::CircularArrayBuffer{T,3}, inds::Vector{Int}, n::Int) where {T} = reshape(view(cb.buffer, :, :, [_buffer_frame(cb, i) for x in inds for i in x:x+n-1]), size(cb.buffer, 1), size(cb.buffer, 2), n, length(inds))
consecutive_view(cb::CircularArrayBuffer{T,4}, inds::Vector{Int}, n::Int) where {T} = reshape(view(cb.buffer, :, :, :, [_buffer_frame(cb, i) for x in inds for i in x:x+n-1]), size(cb.buffer, 1), size(cb.buffer, 2), size(cb.buffer, 3), n, length(inds))


consecutive_view(cb::CircularArrayBuffer{T,1}, inds::Vector{Int}, n::Int, c::Int) where {T} = reshape(view(cb.buffer, [_buffer_frame(cb, c) for x in inds for i in x:x+n-1 for c in i-c+1:i]), c, n, length(inds))
consecutive_view(cb::CircularArrayBuffer{T,2}, inds::Vector{Int}, n::Int, c::Int) where {T} = reshape(view(cb.buffer, :, [_buffer_frame(cb, c) for x in inds for i in x:x+n-1 for c in i-c+1:i]), size(cb.buffer, 1), c, n, length(inds))
consecutive_view(cb::CircularArrayBuffer{T,3}, inds::Vector{Int}, n::Int, c::Int) where {T} = reshape(view(cb.buffer, :, :, [_buffer_frame(cb, c) for x in inds for i in x:x+n-1 for c in i-c+1:i]), size(cb.buffer, 1), size(cb.buffer, 2), c, n, length(inds))
consecutive_view(cb::CircularArrayBuffer{T,4}, inds::Vector{Int}, n::Int, c::Int) where {T} = reshape(view(cb.buffer, :, :, :, [_buffer_frame(cb, c) for x in inds for i in x:x+n-1 for c in i-c+1:i]), size(cb.buffer, 1), size(cb.buffer, 2), size(cb.buffer, 3), c, n, length(inds))

consecutive_view(cb::CircularArrayBuffer, inds::Vector{Int}, n::Int, ::Nothing) = consecutive_view(cb, inds, n)
consecutive_view(cb::CircularArrayBuffer, inds::Vector{Int}, ::Nothing, ::Nothing) = consecutive_view(cb, inds)

Base.getindex(cb::CircularArrayBuffer{T, 2}, i1::Int, i2::Int) where T = cb.buffer[i1, _buffer_frame(cb, i2)]
Base.getindex(cb::CircularArrayBuffer{T, 3}, i1::Int, i2::Int, i3::Int) where T = cb.buffer[i1, i2, _buffer_frame(cb, i3)]
Base.getindex(cb::CircularArrayBuffer{T, 4}, i1::Int, i2::Int, i3::Int, i4::Int) where T = cb.buffer[i1, i2, i3, _buffer_frame(cb, i4)]