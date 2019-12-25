export AlternateExplorer

using Flux

"""
    AlternateExplorer(n::Int)
Used to ensure that all actions are selected alternatively.
# Fields
- `n::Int`: means the optional actions are `1:n`.
- `step::Int=0`: record the number of times that the selector is applied.
"""
Base.@kwdef mutable struct AlternateExplorer <: AbstractExplorer
    n::Int
    step::Int = 1
end

Base.copy(p::AlternateExplorer) = AlternateExplorer(p.n, p.step)

"""
    (s::AlternateExplorer)(values::Any)
Here the `values` is ignored. The returned action is based on the `step` of `s`.
# Example
```julia
julia> selector = AlternateExplorer(n=3)
AlternateExplorer(3, 0)
julia> any_state = 0 # for AlternateExplorer, the value of actions can be anything
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
function (s::AlternateExplorer)(values)
    res = (s.step - 1) % s.n + 1
    s.step += 1
    res
end

RLBase.get_distribution(s::AlternateExplorer, values) = Flux.OneHotVector((s.step - 1) % s.n + 1, s.n)

RLBase.reset!(s::AlternateExplorer) = s.step=1