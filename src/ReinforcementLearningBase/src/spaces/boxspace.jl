"A box in R^n."
struct BoxSpace{T <: Number,N} <: AbstractSpace
    low::Array{T,N}
    high::Array{T,N}
end

size(s::BoxSpace) = size(s.low)

"""
    BoxSpace(low::Number, high::Number, size::Tuple{Vararg{Int}}=(1,))

```julia
BoxSpace(-1, 1)
BoxSpace(-1, 1, (2,3))
```
"""
BoxSpace(low::Number, high::Number, size::Tuple{Vararg{Int}}=(1,)) = BoxSpace(fill(low, size), fill(high, size))

"""
    BoxSpace(low::Array{<:Number}, high::Array{<:Number})

```julia
BoxSpace([0, 0, 0], [1, 2, 3])
```
"""
BoxSpace(low::Array{<:Number}, high::Array{<:Number}) = size(low) == size(high) && BoxSpace(low, high)

==(x::BoxSpace, y::BoxSpace) = x.low == y.low && x.high == y.high

sample(s::BoxSpace) = map((l, h) -> l + rand() * (h - l), s.low, s.high)

occursin(xs::Array{<:Number}, s::BoxSpace) = size(s) == size(xs) && 
    all(map((a, b, c) -> a ≤ b ≤ c, s.low, xs, s.high))
occursin(x::Number, s::BoxSpace) = all(map((l, h) -> l ≤ x ≤ h, s.low, s.high))