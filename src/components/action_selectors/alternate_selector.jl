export AlternateSelector

"""
    AlternateSelector(n::Int)

Used to ensure that all actions are selected alternatively.

# Fields

- `n::Int`: means the optional actions are `1:n`.
- `step::Int=0`: record the number of times that the selector is applied.
"""
Base.@kwdef mutable struct AlternateSelector <: AbstractDiscreteActionSelector
    n::Int
    step::Int = 0
end

"""
    (s::AlternateSelector)(values::Any)

Here the `values` is ignored. The returned action is based on the `step` of `s`.

# Example

```julia
julia> selector = AlternateSelector(n=3)
AlternateSelector(3, 0)

julia> any_state = 0 # for AlternateSelector, the value of actions can be anything

julia> [selector(any_state) for i in 1:10]  # iterate through all actions
10-element Array{Int64,1}:
 1
 2
 3
 1
 ⋮
 1
 2
 3
 1
```
"""
function (s::AlternateSelector)(values::Any; kw...)
    s.step += 1
    (s.step - 1) % s.n + 1
end