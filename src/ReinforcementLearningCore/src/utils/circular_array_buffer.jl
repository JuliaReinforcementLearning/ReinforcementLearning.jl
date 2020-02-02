export CircularArrayBuffer, capacity, isfull, nframes

"""
    CircularArrayBuffer{T}(d::Integer...) -> CircularArrayBuffer{T, N}

`CircularArrayBuffer` uses a `N`-dimention `Array` of size `d` to serve as a buffer for
`N-1`-dimention `Array`s with the same size.

# Examples

```julia-repl
julia> b = CircularArrayBuffer{Float64}(2, 2, 3)
2×2×0 CircularArrayBuffer{Float64,3}

julia> capacity(b)
3

julia> length(b)
0

julia> push!(b, [1. 1.; 2. 2.])
2×2×1 CircularArrayBuffer{Float64,3}:
[:, :, 1] =
 1.0  1.0
 2.0  2.0

julia> b
2×2×1 CircularArrayBuffer{Float64,3}:
[:, :, 1] =
 1.0  1.0
 2.0  2.0

julia> length(b)
4

julia> nframes(cb::CircularArrayBuffer) = cb.length
nframes (generic function with 1 method)

julia> nframes(b)
1

julia> ones(2,2)
2×2 Array{Float64,2}:
 1.0  1.0
 1.0  1.0

julia> 3 .* ones(2,2)
2×2 Array{Float64,2}:
 3.0  3.0
 3.0  3.0

julia> 3 * ones(2,2)
2×2 Array{Float64,2}:
 3.0  3.0
 3.0  3.0

julia> b = CircularArrayBuffer{Float64}(2, 2, 3)
2×2×0 CircularArrayBuffer{Float64,3}

julia> capacity(b)
3

julia> nframes(b)
0

julia> push!(b, 1 .* ones(2,2))
2×2×1 CircularArrayBuffer{Float64,3}:
[:, :, 1] =
 1.0  1.0
 1.0  1.0

julia> b
2×2×1 CircularArrayBuffer{Float64,3}:
[:, :, 1] =
 1.0  1.0
 1.0  1.0

julia> nframes(b)
1

julia> for i in 2:4
           push!(b, i .* ones(2,2))
       end

julia> b
2×2×3 CircularArrayBuffer{Float64,3}:
[:, :, 1] =
 2.0  2.0
 2.0  2.0

[:, :, 2] =
 3.0  3.0
 3.0  3.0

[:, :, 3] =
 4.0  4.0
 4.0  4.0

julia> isfull(b)
true

julia> nframes(b)
3

julia> size(b)
(2, 2, 3)
```
"""
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
Base.size(cb::CircularArrayBuffer{<:Any,N}, i::Integer) where {N} =
    i == N ? cb.length : size(cb.buffer, i)
Base.size(cb::CircularArrayBuffer{<:Any,N}) where {N} = ntuple(M -> size(cb, M), N)
Base.getindex(cb::CircularArrayBuffer{T,N}, i::Int) where {T,N} =
    getindex(cb.buffer, _buffer_index(cb, i))
Base.setindex!(cb::CircularArrayBuffer{T,N}, v, i::Int) where {T,N} =
    setindex!(cb.buffer, v, _buffer_index(cb, i))
capacity(cb::CircularArrayBuffer{T,N}) where {T,N} = size(cb.buffer, N)
nframes(cb::CircularArrayBuffer) = cb.length
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

"""
    update!(cb::CircularArrayBuffer{T,N}, data::AbstractArray{T})

`update!` the last frame of `cb` with data.
"""
function RLBase.update!(cb::CircularArrayBuffer{T,N}, data::AbstractArray{T}) where {T,N}
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

function Base.push!(cb::CircularArrayBuffer{T,N}, ::Missing) where {T,N}
    if cb.length == capacity(cb)
        cb.first = (cb.first == capacity(cb) ? 1 : cb.first + 1)
    else
        cb.length += 1
    end
    cb
end

function Base.pop!(cb::CircularArrayBuffer)
    res = select_last_frame(cb)
    if cb.length <= 0
        throw(ArgumentError("buffer must be non-empty"))
    else
        cb.length -= 1
    end
    res
end

# #####
# # gpu related
# ####

# if has_cuda()
#     using CuArrays
#     import Adapt:adapt

#     adapt(T::Type{<:CuArray}, x::SubArray{<:Any, <:Any, <:CircularArrayBuffer}) = T(x)
# end
