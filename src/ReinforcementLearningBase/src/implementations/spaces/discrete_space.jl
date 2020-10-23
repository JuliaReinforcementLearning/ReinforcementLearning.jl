export DiscreteSpace, ActionProbPair

using Random

"""
    DiscreteSpace(span)

The `span` can be of any iterators.

# Example

```julia-repl
julia > s = DiscreteSpace([1, 2, 3])
DiscreteSpace{Array{Int64,1}}([1, 2, 3])

julia > 0 ∉ s
true

julia > 2 ∈ s
true

julia > s = DiscreteSpace(Set([:a, :c, :a, :b]))
DiscreteSpace{Set{Symbol}}(Set(Symbol[:a, :b, :c]))

julia > s = DiscreteSpace(3)
DiscreteSpace{UnitRange{Int64}}(1:3)
```
"""
struct DiscreteSpace{T} <: AbstractSpace
    span::T
end

"""
    DiscreteSpace(high::T)

Create a `DiscreteSpace` with span of `1:high`
"""
DiscreteSpace(high::T) where {T<:Integer} = DiscreteSpace(one(T), high)

function DiscreteSpace(low::T, high::T) where {T<:Integer}
    high >= low || throw(ArgumentError("$high must be >= $low"))
    return DiscreteSpace(low:high)
end

"""
    DiscreteSpace(low, high)

Create a `DiscreteSpace` with span of `low:high`
"""
DiscreteSpace(low, high) = DiscreteSpace(promote(low, high)...)

Base.eltype(s::DiscreteSpace) = eltype(s.span)
Base.in(x, s::DiscreteSpace) = x in s.span
Random.rand(rng::AbstractRNG, s::DiscreteSpace) = rand(rng, s.span)

Base.length(s::DiscreteSpace) = length(s.span)

Base.convert(
    ::Type{AbstractSpace},
    s::Union{<:Integer,<:UnitRange,<:Vector,<:Tuple,<:Set},
) = DiscreteSpace(s)

Base.iterate(s::DiscreteSpace, args...) = iterate(s.span, args...)
Base.getindex(s::DiscreteSpace, args...) = getindex(s.span, args...)

#####
# ActionWithProb
#####

struct ActionProbPair{A,P}
    action::A
    prob::P
end

Random.rand(rng::AbstractRNG, s::AbstractVector{<:ActionProbPair}) =
    s[weighted_sample(rng, (x.prob for x in s))]

(env::AbstractEnv)(a::ActionProbPair) = env(a.action)
