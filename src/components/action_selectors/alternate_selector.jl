export AlternateSelector

"""
    AlternateSelector <: AbstractDiscreteActionSelector

Used to ensure that all actions are selected alternatively.

    AlternateSelector(n::Int)

`n::Int` means the optional actions are `1:n`.
"""
struct AlternateSelector <: AbstractDiscreteActionSelector
    n::Int
end

"""
    (s::AlternateSelector)(values::Any; step)

`step` must start with `1`.
Ignore the action `values`, generate an action alternatively.

## Example

```julia
julia> selector = AlternateSelector(3)
AlternateSelector(3, 0)

julia> any_state = 0 # for AlternateSelector, state can be anything

julia> [selector(any_state) for i in 1:10]  # iterate through all actions
10-element Array{Int64,1}:
 1
 2
 3
 1
 2
 3
 1
 2
 3
 1
```
"""
(s::AlternateSelector)(values::Any; step, kw...) = (step - 1) % s.n + 1