export EpsilonGreedyExplorer, GreedyExplorer

using Random
using Distributions: Categorical
using Flux: onehot

"""
    EpsilonGreedyExplorer{T}(;kwargs...)
    EpsilonGreedyExplorer(ϵ) -> EpsilonGreedyExplorer{:linear}(; ϵ_stable = ϵ)

> Epsilon-greedy strategy: The best lever is selected for a proportion `1 - epsilon` of the trials, and a lever is selected at random (with uniform probability) for a proportion epsilon . [Multi-armed_bandit](https://en.wikipedia.org/wiki/Multi-armed_bandit)

Two kinds of epsilon-decreasing strategy are implemented here (`linear` and `exp`).

> Epsilon-decreasing strategy: Similar to the epsilon-greedy strategy, except that the value of epsilon decreases as the experiment progresses, resulting in highly explorative behaviour at the start and highly exploitative behaviour at the finish.  - [Multi-armed_bandit](https://en.wikipedia.org/wiki/Multi-armed_bandit)

# Keywords

- `T::Symbol`: defines how to calculate the epsilon in the warmup steps. Supported values are `linear` and `exp`.
- `step::Int = 1`: record the current step.
- `ϵ_init::Float64 = 1.0`: initial epsilon.
- `warmup_steps::Int=0`: the number of steps to use `ϵ_init`.
- `decay_steps::Int=0`: the number of steps for epsilon to decay from `ϵ_init` to `ϵ_stable`.
- `ϵ_stable::Float64`: the epsilon after `warmup_steps + decay_steps`.
- `is_break_tie=false`: randomly select an action of the same maximum values if set to `true`.
- `rng=Random.default_rng()`: set the internal RNG.

# Example

```julia
s_lin = EpsilonGreedyExplorer(kind=:linear, ϵ_init=0.9, ϵ_stable=0.1, warmup_steps=100, decay_steps=100)
plot([RLCore.get_ϵ(s_lin, i) for i in 1:500], label="linear epsilon")
s_exp = EpsilonGreedyExplorer(kind=:exp, ϵ_init=0.9, ϵ_stable=0.1, warmup_steps=100, decay_steps=100)
plot!([RLCore.get_ϵ(s_exp, i) for i in 1:500], label="exp epsilon")
```
![](https://github.com/JuliaReinforcementLearning/ReinforcementLearning.jl/raw/main/docs/src/assets/epsilon_greedy_selector.png)
"""
mutable struct EpsilonGreedyExplorer{Kind,IsBreakTie,R} <: AbstractExplorer
    ϵ_stable::Float64
    ϵ_init::Float64
    warmup_steps::Int
    decay_steps::Int
    step::Int
    rng::R
end

function EpsilonGreedyExplorer(;
    ϵ_stable,
    kind=:linear,
    ϵ_init=1.0,
    warmup_steps=0,
    decay_steps=0,
    step=1,
    is_break_tie=false,
    rng=Random.default_rng()
)
    EpsilonGreedyExplorer{kind,is_break_tie,typeof(rng)}(
        ϵ_stable,
        ϵ_init,
        warmup_steps,
        decay_steps,
        step,
        rng,
    )
end

EpsilonGreedyExplorer(ϵ; kwargs...) = EpsilonGreedyExplorer(; ϵ_stable=ϵ, kwargs...)

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
    RLBase.plan!(s::EpsilonGreedyExplorer, values; step) where T

!!! note
    If multiple values with the same maximum value are found.
    Then a random one will be returned when `is_break_tie==true`.

    `NaN` will be filtered unless all the values are `NaN`.
    In that case, a random one will be returned.
"""
function RLBase.plan!(s::EpsilonGreedyExplorer{<:Any,true}, values::A) where {I<:Real, A<:Union{Vector{I}, SubArray{I}}}
    ϵ = get_ϵ(s)
    s.step += 1
    rand(s.rng) >= ϵ ? rand(s.rng, find_all_max(values)[2]) : rand(s.rng, 1:length(values))
end

function RLBase.plan!(s::EpsilonGreedyExplorer{<:Any,false}, values::A) where {I<:Real, A<:Union{Vector{I}, SubArray{I}}}
    ϵ = get_ϵ(s)
    s.step += 1
    rand(s.rng) >= ϵ ? findmax(values)[2] : rand(s.rng, 1:length(values))
end

#####

RLBase.plan!(s::EpsilonGreedyExplorer{<:Any,true}, x::A, mask::Trues) where {I<:Real, A<:Union{Vector{I}, SubArray{I}}} = RLBase.plan!(s, x)

function RLBase.plan!(s::EpsilonGreedyExplorer{<:Any,true}, values::A, mask::M) where {I<:Real, A<:Union{Vector{I}, SubArray{I}}, M<:Union{BitVector, Vector{Bool}}}
    ϵ = get_ϵ(s)
    s.step += 1
    rand(s.rng) >= ϵ ? rand(s.rng, find_all_max(values, mask)[2]) :
    rand(s.rng, findall(mask))
end

RLBase.plan!(s::EpsilonGreedyExplorer{<:Any,false}, x::A, mask::Trues) where{I<:Real, A<:Union{Vector{I}, SubArray{I}}} = RLBase.plan!(s, x)

function RLBase.plan!(s::EpsilonGreedyExplorer{<:Any,false}, values::A, mask::M) where {I<:Real, A<:Union{Vector{I}, SubArray{I}}, M<:Union{BitVector, Vector{Bool}}}
    ϵ = get_ϵ(s)
    s.step += 1
    rand(s.rng) >= ϵ ? findmax_masked(values, mask)[2] : rand(s.rng, findall(mask))
end

#####

"""
    prob(s::EpsilonGreedyExplorer, values) ->Categorical
    prob(s::EpsilonGreedyExplorer, values, mask) ->Categorical

Return the probability of selecting each action given the estimated `values` of each action.
"""
function RLBase.prob(s::EpsilonGreedyExplorer{<:Any,true}, values::A) where {I<:Real, A<:Union{Vector{I}, SubArray{I}}}
    ϵ, n = get_ϵ(s), length(values)
    probs = fill(ϵ / n, n)
    max_val_inds = find_all_max(values)[2]
    for ind in max_val_inds
        probs[ind] += (1 - ϵ) / length(max_val_inds)
    end
    Categorical(probs; check_args=false)
end

function RLBase.prob(s::EpsilonGreedyExplorer{<:Any,true}, values, action::Integer)
    ϵ, n = get_ϵ(s), length(values)
    max_val_inds = find_all_max(values)[2]
    if action in max_val_inds
        ϵ / n + (1 - ϵ) / length(max_val_inds)
    else
        ϵ / n
    end
end

function RLBase.prob(s::EpsilonGreedyExplorer{<:Any,false}, values)
    ϵ, n = get_ϵ(s), length(values)
    probs = fill(ϵ / n, n)
    probs[findmax(values)[2]] += 1 - ϵ
    Categorical(probs; check_args=false)
end

function RLBase.prob(s::EpsilonGreedyExplorer{<:Any,false}, values, action::Integer)
    ϵ, n = get_ϵ(s), length(values)
    if action == findmax(values)[2]
        ϵ / n + 1 - ϵ
    else
        ϵ / n
    end
end

function RLBase.prob(s::EpsilonGreedyExplorer{<:Any,true}, values, mask)
    ϵ, n = get_ϵ(s), length(values)
    probs = zeros(n)
    probs[mask] .= ϵ / sum(mask)
    max_val_inds = find_all_max(values, mask)[2]
    for ind in max_val_inds
        probs[ind] += (1 - ϵ) / length(max_val_inds)
    end
    Categorical(probs; check_args=false)
end

function RLBase.prob(s::EpsilonGreedyExplorer{<:Any,false}, values, mask)
    ϵ, n = get_ϵ(s), length(values)
    probs = zeros(n)
    probs[mask] .= ϵ / sum(mask)
    probs[findmax_masked(values, mask)[2]] += 1 - ϵ
    Categorical(probs; check_args=false)
end

#####

# Though we can achieve the same goal by setting the ϵ of [`EpsilonGreedyExplorer`](@ref) to 0,
# the GreedyExplorer is much faster.
struct GreedyExplorer <: AbstractExplorer end

RLBase.plan!(s::GreedyExplorer, x, mask::Trues) = s(x)

RLBase.plan!(s::GreedyExplorer, values) = findmax(values)[2]
RLBase.plan!(s::GreedyExplorer, values, mask) = findmax_masked(values, mask)[2]

RLBase.prob(s::GreedyExplorer, values) =
    Categorical(onehot(findmax(values)[2], 1:length(values)); check_args=false)

RLBase.prob(s::GreedyExplorer, values, action::Integer) =
    findmax(values)[2] == action ? 1.0 : 0.0

RLBase.prob(s::GreedyExplorer, values, mask) =
    Categorical(onehot(findmax_masked(values, mask)[2], length(values)); check_args=false)
