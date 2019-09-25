export Tiling, encode

import Base: length, -

"""
    Tiling(ranges::NTuple{N, Tr}) where {N, Tr}

Using a tuple of `ranges` to simulate a tiling.
The length of `ranges` indicates the dimension of tilling.

# Example
```julia
julia> t = Tiling((1:2:5, 10:5:20))
Tiling{2,StepRange{Int64,Int64}}((1:2:5, 10:5:20), [1 3; 2 4])

julia> encode(t, (2, 12))  # encode into an Int
1

julia> encode(t, (2, 18))
3

julia> t2 = t - (1, 3)  # shift a little to get a new Tiling
Tiling{2,StepRange{Int64,Int64}}((0:2:4, 7:5:17), [1 3; 2 4])
```
"""
struct Tiling{N,Tr<:AbstractRange}
    ranges::NTuple{N,Tr}
    inds::LinearIndices{N,NTuple{N,Base.OneTo{Int}}}
    Tiling(ranges::NTuple{N,Tr}) where {N,Tr} =
        new{N,Tr}(ranges, LinearIndices(Tuple(length(r) - 1 for r in ranges)))
end

"""
    (-)(t::Tiling, xs)

Shift `t` along each dimension by each element in `xs`.
"""
function Base.:-(t::Tiling, xs)
    Tiling(Tuple(r .- x for (r, x) in zip(t.ranges, xs)))
end

Base.length(t::Tiling) = reduce(*, (length(r) - 1 for r in t.ranges))

encode(range::AbstractRange, x) = floor(Int, div(x - range[1], step(range)) + 1)


# TODO: use @generator here!
encode(t::Tiling{1}, x::Number) = encode(t.ranges[1], x)
encode(t::Tiling{1}, xs) = encode(t.ranges[1], xs[1])
encode(t::Tiling{2}, xs) =
    t.inds[CartesianIndex(encode(t.ranges[1], xs[1]), encode(t.ranges[2], xs[2]))]
encode(t::Tiling{3}, xs) =
    t.inds[CartesianIndex(
        encode(t.ranges[1], xs[1]),
        encode(t.ranges[2], xs[2]),
        encode(t.ranges[3], xs[3]),
    )]