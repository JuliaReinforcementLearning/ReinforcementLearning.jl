export EpsilonGreedySelector

using StatsBase:sample
using .Utils:findallmax

"""
    EpsilonGreedySelector <: AbstractDiscreteActionSelector
    EpsilonGreedySelector(ϵ; ϵ_init=1.0, warmup_steps=0, decay_steps=0, decay_method=:linear)
"""
struct EpsilonGreedySelector{T} <: AbstractDiscreteActionSelector
    ϵ_init::Float64
    ϵ_stable::Float64
    warmup_steps::Int
    decay_steps::Int

    function EpsilonGreedySelector(ϵ_stable; ϵ_init=1.0, warmup_steps=0, decay_steps=0, decay_method=:linear)
        new{decay_method}(ϵ_init, ϵ_stable, warmup_steps, decay_steps)
    end
end

function get_ϵ(s::EpsilonGreedySelector{:linear}, step)
    if step <= s.warmup_steps
        s.ϵ_init
    elseif step >= (s.warmup_steps + s.decay_steps)
        s.ϵ_stable
    else
        steps_left = s.warmup_steps + s.decay_steps - step
        s.ϵ_stable + steps_left / s.decay_steps * (s.ϵ_init - s.ϵ_stable)
    end
end

function get_ϵ(s::EpsilonGreedySelector{:exp}, step)
    if step <= s.warmup_steps
        s.ϵ_init
    else
        n = step - s.warmup_steps
        s.ϵ_stable + (s.ϵ_init - s.ϵ_stable) * exp(-1. * n / s.decay_steps)
    end
end

"""
    (s::EpsilonGreedySelector)(values; step) where T

!!! note
    If multiple values with the same maximum value are found.
    Then a random one will be returned!

    `NaN` will be filtered unless all the values are `NaN`.
    In that case, a random one will be returned.
"""
function (s::EpsilonGreedySelector)(values; step, kw...)
    ϵ = get_ϵ(s, step)
    rand() > ϵ ? sample(findallmax(values)[2]) : rand(1:length(values))
end