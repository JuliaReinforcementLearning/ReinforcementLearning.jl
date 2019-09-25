export UCBSelector

using .Utils: findallmax

"""
    UCBSelector <: AbstractDiscreteActionSelector
    UCBSelector(na; c=2.0, ϵ=1e-10)

# Arguments
- `na` is the number of actions used to create a internal counter.
- `t` is used to store current time step.
- `c` is used to control the degree of exploration.
"""
mutable struct UCBSelector <: AbstractDiscreteActionSelector
    c::Float64
    actioncounts::Vector{Float64}
    step::Int
    UCBSelector(na; c = 2.0, ϵ = 1e-10, step = 1) = new(c, fill(ϵ, na), 1)
end

@doc raw"""
    (ucb::UCBSelector)(values::AbstractArray)

Unlike [`EpsilonGreedySelector`](@ref), uncertaintyies are considered in UCB.

!!! note
    If multiple values with the same maximum value are found.
    Then a random one will be returned!

```math
A_t = \underset{a}{\arg \max} \left[ Q_t(a) + c \sqrt{\frac{\ln t}{N_t(a)}} \right]
```

See more details at Section (2.7) on Page 35 of the book *Sutton, Richard S., and Andrew G. Barto. Reinforcement learning: An introduction. MIT press, 2018.*
""" function (p::UCBSelector)(values::AbstractArray)
    action = findallmax(@. values + p.c * sqrt(log(p.step + 1) / p.actioncounts))[2] |>
             sample
    p.actioncounts[action] += 1
    p.step += 1
    action
end