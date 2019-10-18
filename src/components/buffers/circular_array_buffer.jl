export CircularArrayBuffer, capacity, isfull, consecutive_view

import Base: view, getindex

"""
    CircularArrayBuffer{E, T, N}

Using a `N` dimension Array to simulate a circular buffer of `N-1` dimensional elements.
Here `E` is the type of element and `T` is same with the `eltype` of `E`.
Call `eltype(b::CircularArrayBuffer{E,T,N})` will return `E`.

# Examples

```julia-repl
julia> b = CircularArrayBuffer{Float64}(2)
0-element CircularArrayBuffer{Float64,Float64,1}

julia> push!(b, rand())
1-element CircularArrayBuffer{Float64,Float64,1}:
 0.9709012378596478

julia> push!(b, rand())
2-element CircularArrayBuffer{Float64,Float64,1}:
 0.9709012378596478
 0.4510778027035365

julia> push!(b, rand())
2-element CircularArrayBuffer{Float64,Float64,1}:
 0.4510778027035365
 0.6774251288208646

julia> b = CircularArrayBuffer{Array{Float64,2}}(3, (3,3))
3×3×0 CircularArrayBuffer{Array{Float64,2},Float64,3}

julia> push!(b, randn(3,3))
3×3×1 CircularArrayBuffer{Array{Float64,2},Float64,3}:
[:, :, 1] =
 -0.548592   0.926179  -1.40998
 -0.0888621  0.177208   0.342665
  0.0925987  1.18531    0.962738
```
"""
mutable struct CircularArrayBuffer{E,T,N} <: AbstractArray{T,N}
    buffer::Array{T,N}
    first::Int
    length::Int
    stepsize::Int
    CircularArrayBuffer{T}(capacity::Integer) where {T} =
        new{T,T,1}(Vector{T}(undef, capacity), 1, 0, 1)
    function CircularArrayBuffer{T}(
        capacity::Integer,
        element_size::Vararg{<:Integer,N},
    ) where {T<:AbstractArray,N}
        ndims(T) == N || throw(DimensionMismatch("the ndims of the specified type $T doesn't math the length of element_size $element_size"))
        new{T,eltype(T),N + 1}(
            Array{eltype(T),N + 1}(undef, element_size..., capacity),
            1,
            0,
            *(element_size...),
        )
    end
    CircularArrayBuffer{T}(
        capacity::Int,
        element_size::Tuple{Vararg{<:Integer,N}},
    ) where {T<:AbstractArray,N} = CircularArrayBuffer{T}(capacity, element_size...)
end

Base.eltype(cb::CircularArrayBuffer{E,T,N}) where {E,T,N} = E
Base.size(cb::CircularArrayBuffer{E,T,N}) where {E,T,N} =
    (size(cb.buffer)[1:N-1]..., cb.length)

# TODO: simplify code bellow
for func in [:view, :getindex]
    @eval @__MODULE__() begin
        $func(cb::CircularArrayBuffer{E,T,1}, i::Int) where {E,T} =
            $func(cb.buffer, _buffer_index(cb, i))
        $func(cb::CircularArrayBuffer{E,T,2}, i::Int) where {E,T} =
            $func(cb.buffer, :, _buffer_index(cb, i))
        $func(cb::CircularArrayBuffer{E,T,3}, i::Int) where {E,T} =
            $func(cb.buffer, :, :, _buffer_index(cb, i))
        $func(cb::CircularArrayBuffer{E,T,4}, i::Int) where {E,T} =
            $func(cb.buffer, :, :, :, _buffer_index(cb, i))
        $func(cb::CircularArrayBuffer{E,T,N}, i::Int) where {E,T,N} =
            $func(cb.buffer, [(:) for _ = 1:N-1]..., _buffer_index(cb, i))

        $func(cb::CircularArrayBuffer{E,T,1}, I::Vector{Int}) where {E,T} =
            $func(cb.buffer, map(i -> _buffer_index(cb, i), I))
        $func(cb::CircularArrayBuffer{E,T,2}, I::Vector{Int}) where {E,T} =
            $func(cb.buffer, :, map(i -> _buffer_index(cb, i), I))
        $func(cb::CircularArrayBuffer{E,T,3}, I::Vector{Int}) where {E,T} =
            $func(cb.buffer, :, :, map(i -> _buffer_index(cb, i), I))
        $func(cb::CircularArrayBuffer{E,T,4}, I::Vector{Int}) where {E,T} =
            $func(cb.buffer, :, :, :, map(i -> _buffer_index(cb, i), I))
        $func(cb::CircularArrayBuffer{E,T,N}, I::Vector{Int}) where {E,T,N} =
            $func(cb.buffer, [(:) for _ = 1:N-1]..., map(i -> _buffer_index(cb, i), I))

        $func(cb::CircularArrayBuffer{E,T,1}, i::UnitRange{Int}) where {E,T} =
            $func(cb.buffer, _buffer_index(cb, i))
        $func(cb::CircularArrayBuffer{E,T,2}, i::UnitRange{Int}) where {E,T} =
            $func(cb.buffer, :, _buffer_index(cb, i))
        $func(cb::CircularArrayBuffer{E,T,3}, i::UnitRange{Int}) where {E,T} =
            $func(cb.buffer, :, :, _buffer_index(cb, i))
        $func(cb::CircularArrayBuffer{E,T,4}, i::UnitRange{Int}) where {E,T} =
            $func(cb.buffer, :, :, :, _buffer_index(cb, i))
        $func(cb::CircularArrayBuffer{E,T,N}, i::UnitRange{Int}) where {E,T,N} =
            $func(cb.buffer, [(:) for _ = 1:N-1]..., _buffer_index(cb, i))
    end
end

# TODO: use @generated instead
Base.setindex!(cb::CircularArrayBuffer{E,T,1}, data, i) where {E,T} =
    cb.buffer[_buffer_index(cb, i)] = data
Base.setindex!(cb::CircularArrayBuffer{E,T,2}, data, i) where {E,T} =
    cb.buffer[:, _buffer_index(cb, i)] = data
Base.setindex!(cb::CircularArrayBuffer{E,T,3}, data, i) where {E,T} =
    cb.buffer[:, :, _buffer_index(cb, i)] = data
Base.setindex!(cb::CircularArrayBuffer{E,T,4}, data, i) where {E,T} =
    cb.buffer[:, :, :, _buffer_index(cb, i)] = data
Base.setindex!(cb::CircularArrayBuffer{E,T,N}, data, i) where {E,T,N} =
    cb.buffer[[(:) for _ = 1:N-1]..., _buffer_index(cb, i)] = data

## kept only to show elements
## never manually get/set the inner elements because it is very slow
## see https://discourse.julialang.org/t/varargs-performance/13578
Base.getindex(cb::CircularArrayBuffer{E,T,N}, I::Vararg{Int,N}) where {E,T,N} =
    getindex(cb.buffer, I[1:N-1]..., _buffer_index(cb, I[end]))

capacity(cb::CircularArrayBuffer) = size(cb.buffer)[end]
Base.length(cb::CircularArrayBuffer) = cb.length
isfull(cb::CircularArrayBuffer) = length(cb) == capacity(cb)
Base.isempty(cb::CircularArrayBuffer) = length(cb) == 0
Base.empty!(cb::CircularArrayBuffer) = (cb.length = 0; cb)

"""
    push!(cb::CircularArrayBuffer{E, T, N}, data::E) where {E, T, N}

Add an element to the back and overwrite front if full.
Make sure that `length(data) == cb.stepsize`
"""
@inline function Base.push!(
    cb::CircularArrayBuffer{E,T,N},
    data::AbstractArray{T},
) where {E,T,N}
    length(data) == cb.stepsize || throw(DimensionMismatch("the length of buffer's stepsize doesn't match the length of data, $(cb.stepsize) != $(length(data))"))
    # if full, increment and overwrite, otherwise push
    if cb.length == capacity(cb)
        cb.first = (cb.first == capacity(cb) ? 1 : cb.first + 1)
    else
        cb.length += 1
    end
    nxt_idx = _buffer_index(cb, cb.length)
    cb.buffer[cb.stepsize*(nxt_idx-1)+1:cb.stepsize*nxt_idx] = data
    cb
end

@inline function Base.push!(cb::CircularArrayBuffer, data::Number)
    if cb.length == capacity(cb)
        cb.first = (cb.first == capacity(cb) ? 1 : cb.first + 1)
    else
        cb.length += 1
    end
    cb.buffer[_buffer_index(cb, cb.length)] = data
    cb
end

function Base.push!(cb::CircularArrayBuffer, f::Function)
    if cb.length == capacity(cb)
        cb.first = (cb.first == capacity(cb) ? 1 : cb.first + 1)
    else
        cb.length += 1
    end
    x = @view(cb[end])
    f(x)
    x
end

@inline function _buffer_index(cb::CircularArrayBuffer, i::Int)
    n = capacity(cb)
    idx = cb.first + i - 1
    if idx > n
        idx - n
    else
        idx
    end
end

@inline function _buffer_index(cb::CircularArrayBuffer, i::UnitRange{Int})
    start = _buffer_index(cb, i.start)
    stop = _buffer_index(cb, i.stop)
    start ≤ stop ? (start:stop) : vcat(start:capacity(cb), 1:stop)
end

function consecutive_view(b::CircularArrayBuffer{E,T,N}, inds, n) where {E,T,N}
    expanded_inds = collect(Iterators.flatten(x:x+n-1 for x in inds))
    reshape(view(b, expanded_inds), size(b.buffer)[1:N-1]..., n, length(inds))
end