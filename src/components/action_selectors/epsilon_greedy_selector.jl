export EpsilonGreedySelector

using StatsBase: sample
using .Utils: findallmax

Base.@kwdef mutable struct EpsilonGreedySelector{T} <: AbstractDiscreteActionSelector
    ϵ_stable::Float64
    ϵ_init::Float64 = 1.0
    warmup_steps::Int = 0
    decay_steps::Int = 0
    step::Int = 1
end

EpsilonGreedySelector(ϵ) = EpsilonGreedySelector{:linear}(; ϵ_stable = ϵ)

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
        s.ϵ_stable + (s.ϵ_init - s.ϵ_stable) * exp(-1.0 * n / s.decay_steps)
    end
end

get_ϵ(s::EpsilonGreedySelector) = get_ϵ(s, s.step)

"""
    (s::EpsilonGreedySelector)(values; step) where T

!!! note
    If multiple values with the same maximum value are found.
    Then a random one will be returned!

    `NaN` will be filtered unless all the values are `NaN`.
    In that case, a random one will be returned.
"""
function (s::EpsilonGreedySelector)(values)
    ϵ = get_ϵ(s)
    s.step += 1
    rand() > ϵ ? sample(findallmax(values)[2]) : rand(1:length(values))
end

function get_prob(s::EpsilonGreedySelector, values)
    ϵ, n = get_ϵ(s), length(values)
    probs = fill(ϵ / n, n)
    max_val_inds = findallmax(values)[2]
    for ind in max_val_inds
        probs[ind] += (1 - ϵ) / length(max_val_inds)
    end
    probs
end