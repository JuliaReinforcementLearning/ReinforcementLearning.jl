export EpsilonGreedyExplorer, GreedyExplorer

using Random
using Distributions: DiscreteNonParametric

"""
    EpsilonGreedyExplorer{T}(;kwargs...)
    EpsilonGreedyExplorer(ϵ) -> EpsilonGreedyExplorer{:linear}(; ϵ_stable = ϵ)

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
s = EpsilonGreedyExplorer{:linear}(ϵ_init=0.9, ϵ_stable=0.1, warmup_steps=100, decay_steps=100)
plot([RL.get_ϵ(s, i) for i in 1:500], label="linear epsilon")
```
![](../assets/img/linear_epsilon_greedy_selector.png)

```julia
s = EpsilonGreedyExplorer{:exp}(ϵ_init=0.9, ϵ_stable=0.1, warmup_steps=100, decay_steps=100)
plot([RL.get_ϵ(s, i) for i in 1:500], label="exp epsilon")
```
![](../assets/img/exp_epsilon_greedy_selector.png)
"""
mutable struct EpsilonGreedyExplorer{Kind,IsBreakTie,R} <: AbstractExplorer
    ϵ_stable::Float64
    ϵ_init::Float64
    warmup_steps::Int
    decay_steps::Int
    step::Int
    rng::R
end

function Base.copy(p::EpsilonGreedyExplorer{Kind,IsBreakTie,R}) where {Kind,IsBreakTie,R}
    EpsilonGreedyExplorer{Kind,IsBreakTie,R}(
        p.ϵ_stable,
        p.ϵ_init,
        p.warmup_steps,
        p.decay_steps,
        p.step,
        copy(p.rng),
    )
end

function EpsilonGreedyExplorer(;
    ϵ_stable,
    kind = :linear,
    ϵ_init = 1.0,
    warmup_steps = 0,
    decay_steps = 0,
    step = 1,
    is_break_tie = false,
    seed = nothing,
)
    rng = MersenneTwister(seed)
    EpsilonGreedyExplorer{kind,is_break_tie,typeof(rng)}(
        ϵ_stable,
        ϵ_init,
        warmup_steps,
        decay_steps,
        step,
        rng,
    )
end

EpsilonGreedyExplorer(ϵ; kwargs...) = EpsilonGreedyExplorer(; ϵ_stable = ϵ, kwargs...)

function get_ϵ(s::EpsilonGreedyExplorer{:linear}, step)
    if step <= s.warmup_steps
        s.ϵ_init
    elseif step >= (s.warmup_steps + s.decay_steps)
        s.ϵ_stable
    else
        steps_left = s.warmup_steps + s.decay_steps - step
        s.ϵ_stable + steps_left / s.decay_steps * (s.ϵ_init - s.ϵ_stable)
    end
end

function get_ϵ(s::EpsilonGreedyExplorer{:exp}, step)
    if step <= s.warmup_steps
        s.ϵ_init
    else
        n = step - s.warmup_steps
        scale = s.ϵ_init - s.ϵ_stable
        s.ϵ_stable + scale * exp(-1.0 * n / s.decay_steps)
    end
end

get_ϵ(s::EpsilonGreedyExplorer) = get_ϵ(s, s.step)

"""
    (s::EpsilonGreedyExplorer)(values; step) where T

!!! note
    If multiple values with the same maximum value are found.
    Then a random one will be returned!

    `NaN` will be filtered unless all the values are `NaN`.
    In that case, a random one will be returned.
"""
function (s::EpsilonGreedyExplorer{<:Any,true})(values)
    ϵ = get_ϵ(s)
    s.step += 1
    rand(s.rng) >= ϵ ? rand(s.rng, find_all_max(values)[2]) : rand(s.rng, 1:length(values))
end

function (s::EpsilonGreedyExplorer{<:Any,false})(values)
    ϵ = get_ϵ(s)
    s.step += 1
    rand(s.rng) >= ϵ ? find_max(values)[2] : rand(s.rng, 1:length(values))
end

function (s::EpsilonGreedyExplorer{<:Any,true})(values, mask)
    ϵ = get_ϵ(s)
    s.step += 1
    rand(s.rng) >= ϵ ? rand(s.rng, find_all_max(values, mask)[2]) :
    rand(s.rng, findall(mask))
end

function (s::EpsilonGreedyExplorer{<:Any,false})(values, mask)
    ϵ = get_ϵ(s)
    s.step += 1
    rand(s.rng) >= ϵ ? find_max(values, mask)[2] : rand(s.rng, findall(mask))
end

Random.seed!(s::EpsilonGreedyExplorer, seed) = Random.seed!(s.rng, seed)

"""
    get_prob(s::EpsilonGreedyExplorer, values) -> DiscreteNonParametric
    get_prob(s::EpsilonGreedyExplorer, values, mask) -> DiscreteNonParametric

Return the probability of selecting each action given the estimated `values` of each action.
"""
function RLBase.get_prob(s::EpsilonGreedyExplorer{<:Any,true}, values)
    ϵ, n = get_ϵ(s), length(values)
    probs = fill(ϵ / n, n)
    max_val_inds = find_all_max(values)[2]
    for ind in max_val_inds
        probs[ind] += (1 - ϵ) / length(max_val_inds)
    end
    DiscreteNonParametric(1:length(probs), probs)
end

function RLBase.get_prob(s::EpsilonGreedyExplorer{<:Any,false}, values)
    ϵ, n = get_ϵ(s), length(values)
    probs = fill(ϵ / n, n)
    probs[find_max(values)[2]] += 1 - ϵ
    DiscreteNonParametric(1:length(probs), probs)
end

function RLBase.get_prob(s::EpsilonGreedyExplorer{<:Any,true}, values, mask)
    ϵ, n = get_ϵ(s), length(values)
    probs = zeros(n)
    probs[mask] .= ϵ / sum(mask)
    max_val_inds = find_all_max(values, mask)[2]
    for ind in max_val_inds
        probs[ind] += (1 - ϵ) / length(max_val_inds)
    end
    DiscreteNonParametric(1:length(probs), probs)
end

function RLBase.get_prob(s::EpsilonGreedyExplorer{<:Any,false}, values, mask)
    ϵ, n = get_ϵ(s), length(values)
    probs = zeros(n)
    probs[mask] .= ϵ / sum(mask)
    probs[find_max(values, mask)[2]] += 1 - ϵ
    DiscreteNonParametric(1:length(probs), probs)
end

RLBase.reset!(s::EpsilonGreedyExplorer) = s.step = 1

struct GreedyExplorer <: AbstractExplorer end

(s::GreedyExplorer)(values) = find_max(values)[2]
(s::GreedyExplorer)(values, mask) = find_max(values, mask)[2]

function RLBase.get_prob(s::GreedyExplorer, values)
    prob = zeros(length(values))
    prob[find_max(values)[2]] = 1.0
    DiscreteNonParametric(1:length(prob), prob)
end

function RLBase.get_prob(s::GreedyExplorer, values, mask)
    prob = zeros(length(values))
    prob[find_max(values, mask)[2]] = 1.0
    DiscreteNonParametric(1:length(prob), prob)
end
