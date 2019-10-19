export EpsilonGreedySelector, get_prob

using StatsBase: sample
using .Utils: findallmax

"""
    EpsilonGreedySelector{T}(;kwargs...)

> Epsilon-greedy strategy: The best lever is selected for a proportion `1 - epsilon` of the trials, and a lever is selected at random (with uniform probability) for a proportion epsilon . [Multi-armed_bandit](https://en.wikipedia.org/wiki/Multi-armed_bandit)

Two kinds of epsilon-decreasing strategy are implmented here (`linear` and `exp`).

> Epsilon-decreasing strategy: Similar to the epsilon-greedy strategy, except that the value of epsilon decreases as the experiment progresses, resulting in highly explorative behaviour at the start and highly exploitative behaviour at the finish.  - [Multi-armed_bandit](https://en.wikipedia.org/wiki/Multi-armed_bandit)

# Keywords

- `T::Symbol`: defines how to calculate the epsilon in the warmup steps. Supported values are `linear` and `exp`.
- `step::Int = 1`: record the current step.
- `ϵ_init::Float64 = 1.0`: initial epsilon.
- `warmup_steps::Int=0`: the number of steps to use `ϵ_init`.
- `decay_steps::Int=0`: the number of steps for epsilon to decay from `ϵ_init` to `ϵ_stable`.
- `ϵ_stable::Float64`: the epsilon after `warmup_steps + decay_steps`.

# Example

```julia
s = EpsilonGreedySelector{:linear}(ϵ_init=0.9, ϵ_stable=0.1, warmup_steps=100, decay_steps=100)
plot([RL.get_ϵ(s, i) for i in 1:500], label="linear epsilon")
```
![](../assets/img/linear_epsilon_greedy_selector.png)

```julia
s = EpsilonGreedySelector{:exp}(ϵ_init=0.9, ϵ_stable=0.1, warmup_steps=100, decay_steps=100)
plot([RL.get_ϵ(s, i) for i in 1:500], label="exp epsilon")
```
![](../assets/img/exp_epsilon_greedy_selector.png)
"""
Base.@kwdef mutable struct EpsilonGreedySelector{T} <: AbstractDiscreteActionSelector
    ϵ_stable::Float64
    ϵ_init::Float64 = 1.0
    warmup_steps::Int = 0
    decay_steps::Int = 0
    step::Int = 1
end

"""
    EpsilonGreedySelector(ϵ) -> EpsilonGreedySelector{:linear}(; ϵ_stable = ϵ)
"""
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

"""
    get_prob(s::EpsilonGreedySelector, values)

Return the probability of selecting each action given the estimated `values` of each action.
"""
function get_prob(s::EpsilonGreedySelector, values)
    ϵ, n = get_ϵ(s), length(values)
    probs = fill(ϵ / n, n)
    max_val_inds = findallmax(values)[2]
    for ind in max_val_inds
        probs[ind] += (1 - ϵ) / length(max_val_inds)
    end
    probs
end